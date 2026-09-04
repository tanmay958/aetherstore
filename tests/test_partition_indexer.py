"""Tests for the indexing core, and for what a crash does to it.

Kafka gives at-least-once delivery. Exactly-once delivery is impossible over a
network that can drop the acknowledgement of a completed write, so the
guarantee has to be built as at-least-once plus idempotent work. These tests
are the proof that the second half holds: they kill the indexer between every
pair of writes and assert the replay produces exactly the right index.

No broker is involved. Offsets are just monotonic integers here, which is what
makes the interesting properties testable at all.
"""

import pytest
from failing_store import CrashAfter

from aether.index.manifest import read_manifest
from aether.index.segment import SegmentReader
from aether.storage import LocalStore
from aether.stream.partition import (
    FlushPolicy,
    PartitionIndexer,
    manifest_key,
    segment_key,
)


def docs(n: int, start: int = 0) -> list[dict]:
    return [
        {
            "title": f"widget{i} samsung smartphone",
            "event_type": "view",
            "ts": 1_700_000_000 + i,
            "event_id": f"e_{i:06d}",
        }
        for i in range(start, start + n)
    ]


def feed(indexer: PartitionIndexer, documents: list[dict], first_offset: int = 100):
    for i, doc in enumerate(documents):
        indexer.add(doc, first_offset + i)


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


# --------------------------------------------------------------------------
# naming, which is what makes everything else work
# --------------------------------------------------------------------------


def test_segment_key_is_derived_from_the_offset_range(store):
    indexer = PartitionIndexer(store, 3, policy=FlushPolicy(max_docs=5))
    feed(indexer, docs(5), first_offset=100)
    meta = indexer.flush()
    assert meta.key == segment_key(3, 100, 104)
    assert meta.key == "segments/p3/00000000000000000100-00000000000000000104.seg"


def test_keys_sort_in_offset_order(store):
    """Fixed width, so lexicographic order is offset order."""
    assert segment_key(0, 9, 9) < segment_key(0, 10, 10) < segment_key(0, 100, 100)


def test_each_partition_writes_its_own_manifest(store):
    """One shared manifest would have several writers and lose segments to
    overwrites. One per partition inherits single-writer safety from the
    consumer group instead."""
    for partition in (0, 1):
        indexer = PartitionIndexer(store, partition, policy=FlushPolicy(max_docs=2))
        feed(indexer, docs(2), first_offset=100)
        indexer.flush()

    assert store.exists(manifest_key(0))
    assert store.exists(manifest_key(1))
    assert read_manifest(store, manifest_key(0)).segments[0].key.startswith("segments/p0/")
    assert read_manifest(store, manifest_key(1)).segments[0].key.startswith("segments/p1/")


# --------------------------------------------------------------------------
# flushing and committing
# --------------------------------------------------------------------------


def test_flushes_when_the_document_count_is_reached(store):
    indexer = PartitionIndexer(store, 0, policy=FlushPolicy(max_docs=3))
    for i, doc in enumerate(docs(3)):
        indexer.add(doc, 100 + i)
        assert indexer.should_flush() is (i == 2)


def test_nothing_to_flush_is_not_a_flush(store):
    indexer = PartitionIndexer(store, 0)
    assert indexer.should_flush() is False
    assert indexer.flush() is None


def test_no_offset_is_committable_until_something_is_durable(store):
    """Committing an offset for work that is not in the manifest would lose
    it on a crash, which is the at-most-once failure."""
    indexer = PartitionIndexer(store, 0, policy=FlushPolicy(max_docs=3))
    feed(indexer, docs(3), first_offset=100)
    assert indexer.next_offset is None

    indexer.flush()
    assert indexer.next_offset == 103  # one past the last durable offset


def test_buffered_documents_are_not_searchable(store):
    indexer = PartitionIndexer(store, 0, policy=FlushPolicy(max_docs=100))
    feed(indexer, docs(5), first_offset=100)
    assert read_manifest(store, manifest_key(0)).segments == ()


def test_a_flush_makes_the_batch_searchable(store):
    indexer = PartitionIndexer(store, 0, policy=FlushPolicy(max_docs=5))
    feed(indexer, docs(5), first_offset=100)
    meta = indexer.flush()

    segment = SegmentReader(store, meta.key)
    assert segment.num_docs == 5
    assert segment.search_and("samsung smartphone") == [0, 1, 2, 3, 4]


def test_successive_flushes_accumulate_in_the_manifest(store):
    indexer = PartitionIndexer(store, 0, policy=FlushPolicy(max_docs=2))
    feed(indexer, docs(2, 0), first_offset=100)
    indexer.flush()
    feed(indexer, docs(2, 2), first_offset=102)
    indexer.flush()

    live = read_manifest(store, manifest_key(0))
    assert len(live.segments) == 2
    assert live.docs == 4
    assert live.generation == 2


# --------------------------------------------------------------------------
# idempotence: the property the whole design rests on
# --------------------------------------------------------------------------


def test_replaying_the_same_offsets_rewrites_identical_bytes(store):
    """Had the key been a uuid or a timestamp, every retry would add a second
    copy of the same documents, every one counted twice by BM25, and relevance
    would be quietly and permanently wrong with no error anywhere."""
    first = PartitionIndexer(store, 0, policy=FlushPolicy(max_docs=5))
    feed(first, docs(5), first_offset=100)
    meta = first.flush()
    original = store.get_range(meta.key, 0, meta.bytes)

    replayed = PartitionIndexer(store, 0, policy=FlushPolicy(max_docs=5))
    feed(replayed, docs(5), first_offset=100)
    again = replayed.flush()

    assert again.key == meta.key
    assert store.get_range(again.key, 0, again.bytes) == original


def test_a_replay_does_not_list_the_segment_twice(store):
    for _ in range(3):
        indexer = PartitionIndexer(store, 0, policy=FlushPolicy(max_docs=5))
        feed(indexer, docs(5), first_offset=100)
        indexer.flush()

    live = read_manifest(store, manifest_key(0))
    assert len(live.segments) == 1
    assert live.docs == 5


# --------------------------------------------------------------------------
# crashes, at each step of the flush
# --------------------------------------------------------------------------


def test_crash_before_the_segment_write_leaves_nothing(store):
    crashing = CrashAfter(store, writes=0)
    indexer = PartitionIndexer(crashing, 0, policy=FlushPolicy(max_docs=5))
    feed(indexer, docs(5), first_offset=100)

    with pytest.raises(CrashAfter.Crash):
        indexer.flush()

    assert read_manifest(store, manifest_key(0)).segments == ()
    assert not store.exists(segment_key(0, 100, 104))


def test_crash_after_the_segment_but_before_the_manifest(store):
    """The segment exists but nothing points at it, so no searcher can see it.
    It is invisible garbage, and the replay overwrites it."""
    crashing = CrashAfter(store, writes=1)
    indexer = PartitionIndexer(crashing, 0, policy=FlushPolicy(max_docs=5))
    feed(indexer, docs(5), first_offset=100)

    with pytest.raises(CrashAfter.Crash):
        indexer.flush()

    assert store.exists(segment_key(0, 100, 104))
    assert read_manifest(store, manifest_key(0)).segments == ()

    # Restart: offsets were never committed, so the batch is replayed.
    recovered = PartitionIndexer(store, 0, policy=FlushPolicy(max_docs=5))
    feed(recovered, docs(5), first_offset=100)
    meta = recovered.flush()

    live = read_manifest(store, manifest_key(0))
    assert len(live.segments) == 1
    assert live.docs == 5
    assert SegmentReader(store, meta.key).num_docs == 5


def test_crash_after_the_manifest_but_before_the_commit(store):
    """The data is already searchable and the offset was never committed, so
    the replay redoes work that is already correct. Both writes are
    idempotent, so redoing them changes nothing."""
    indexer = PartitionIndexer(store, 0, policy=FlushPolicy(max_docs=5))
    feed(indexer, docs(5), first_offset=100)
    indexer.flush()
    # The commit never happened; the process died here.

    recovered = PartitionIndexer(store, 0, policy=FlushPolicy(max_docs=5))
    feed(recovered, docs(5), first_offset=100)
    recovered.flush()

    live = read_manifest(store, manifest_key(0))
    assert len(live.segments) == 1
    assert live.docs == 5


def test_a_crash_mid_stream_loses_nothing_and_duplicates_nothing(store):
    """The end to end claim: kill it partway, replay from the last committed
    offset, and the index holds every document exactly once."""
    policy = FlushPolicy(max_docs=4)

    indexer = PartitionIndexer(store, 0, policy=policy)
    feed(indexer, docs(4, 0), first_offset=100)
    indexer.flush()
    committed = indexer.next_offset  # 104

    # Four more arrive and are buffered, then the process dies.
    feed(indexer, docs(4, 4), first_offset=104)
    del indexer

    recovered = PartitionIndexer(store, 0, policy=policy)
    feed(recovered, docs(4, 4), first_offset=committed)
    recovered.flush()

    live = read_manifest(store, manifest_key(0))
    assert live.docs == 8
    seen = [
        SegmentReader(store, meta.key).document(i)["event_id"]
        for meta in live.segments
        for i in range(meta.docs)
    ]
    assert sorted(seen) == [f"e_{i:06d}" for i in range(8)]
    assert len(seen) == len(set(seen))


# --------------------------------------------------------------------------
# rebalancing
# --------------------------------------------------------------------------


def test_discard_drops_buffered_work_without_writing(store):
    """What a revoked partition does. The offsets were never committed, so the
    next owner replays them; flushing here would race that owner on the same
    manifest."""
    indexer = PartitionIndexer(store, 0, policy=FlushPolicy(max_docs=100))
    feed(indexer, docs(5), first_offset=100)

    indexer.discard()

    assert indexer.buffered == 0
    assert read_manifest(store, manifest_key(0)).segments == ()
    assert indexer.flush() is None


def test_a_new_owner_picks_up_the_existing_manifest(store):
    """Partition handover: the inheriting consumer must extend the live set,
    not replace it."""
    first = PartitionIndexer(store, 0, policy=FlushPolicy(max_docs=3))
    feed(first, docs(3, 0), first_offset=100)
    first.flush()

    inheritor = PartitionIndexer(store, 0, policy=FlushPolicy(max_docs=3))
    assert len(inheritor.manifest.segments) == 1

    feed(inheritor, docs(3, 3), first_offset=103)
    inheritor.flush()
    assert read_manifest(store, manifest_key(0)).docs == 6


# --------------------------------------------------------------------------
# the flush policy, and the orphan trap
# --------------------------------------------------------------------------


def test_count_based_flushing_is_deterministic():
    assert FlushPolicy(max_docs=10).deterministic is True


def test_time_based_flushing_is_not(store):
    """A timer makes segment boundaries depend on how busy the machine was, so
    a replay can produce a different key and orphan the first object."""
    assert FlushPolicy(max_docs=10, max_idle_seconds=30).deterministic is False


def test_idle_flushing_seals_a_partial_batch(store):
    now = [1000.0]
    indexer = PartitionIndexer(
        store,
        0,
        policy=FlushPolicy(max_docs=1000, max_idle_seconds=30),
        clock=lambda: now[0],
    )
    feed(indexer, docs(2), first_offset=100)
    assert indexer.should_flush() is False

    now[0] += 31
    assert indexer.should_flush() is True
    assert indexer.flush().docs == 2


def test_a_replay_supersedes_a_partial_flush_instead_of_duplicating_it(store):
    """The trap that a timer-based flush sets, and the fix for it.

    A first attempt seals offsets 100 to 101 because the timer fired, then
    crashes before committing. The replay, on a less busy machine, reaches 104
    before flushing, so it writes a *different* key covering a wider range.

    Appending both to the manifest would list offsets 100 and 101 twice, and
    those documents would be counted twice by BM25 with nothing reporting an
    error. Publishing evicts by offset range rather than by key, so the
    narrower segment is unlisted the moment the wider one supersedes it.
    """
    now = [1000.0]
    early = PartitionIndexer(
        store,
        0,
        policy=FlushPolicy(max_docs=1000, max_idle_seconds=30),
        clock=lambda: now[0],
    )
    feed(early, docs(2, 0), first_offset=100)
    now[0] += 31
    superseded = early.flush()

    # The offset was never committed, so the replay starts again at 100.
    replayed = PartitionIndexer(store, 0, policy=FlushPolicy(max_docs=5))
    feed(replayed, docs(5, 0), first_offset=100)
    live_segment = replayed.flush()

    assert superseded.key != live_segment.key

    live = read_manifest(store, manifest_key(0))
    assert [s.key for s in live.segments] == [live_segment.key]
    assert live.docs == 5  # not 7

    # The object itself remains, unreferenced and invisible. That is the real
    # cost of a non-deterministic flush boundary, and why a collector is
    # needed once --max-idle is enabled.
    assert store.exists(superseded.key)


def test_documents_are_never_indexed_twice_after_a_partial_replay(store):
    """The consequence stated directly: a duplicated document inflates its own
    term frequencies and quietly corrupts every score it participates in."""
    now = [1000.0]
    early = PartitionIndexer(
        store,
        0,
        policy=FlushPolicy(max_docs=1000, max_idle_seconds=30),
        clock=lambda: now[0],
    )
    feed(early, docs(2, 0), first_offset=100)
    now[0] += 31
    early.flush()

    replayed = PartitionIndexer(store, 0, policy=FlushPolicy(max_docs=5))
    feed(replayed, docs(5, 0), first_offset=100)
    replayed.flush()

    live = read_manifest(store, manifest_key(0))
    seen = [
        SegmentReader(store, meta.key).document(i)["event_id"]
        for meta in live.segments
        for i in range(meta.docs)
    ]
    assert sorted(seen) == [f"e_{i:06d}" for i in range(5)]
    assert len(seen) == len(set(seen))


@pytest.mark.parametrize("kwargs", [{"max_docs": 0}, {"max_idle_seconds": 0}])
def test_rejects_a_nonsense_policy(kwargs):
    with pytest.raises(ValueError):
        FlushPolicy(**kwargs)
