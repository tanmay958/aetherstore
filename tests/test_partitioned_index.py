"""Searching an index the streaming indexer produced.

The batch ingester writes one manifest. The streaming indexer writes one per
Kafka partition, because single-writer safety is inherited from the log:
exactly one consumer owns a partition, so exactly one process writes that
partition's manifest and nothing needs locking.

That means a streamed index cannot be searched by reading one object, and
until this existed it could not be searched as a whole at all. The Kafka
integration tests each searched a single partition, so nothing noticed.

The test that matters most here is `test_partitions_do_not_evict_each_other`.
"""

import pytest

from aether.data.rees46 import iter_events
from aether.index.coordinator import Coordinator
from aether.index.manifest import (
    Manifest,
    SegmentMeta,
    read_manifest,
    union_manifests,
    write_manifest,
)
from aether.index.memory import build_index
from aether.index.segment import write_segment
from aether.storage import LocalStore
from aether.stream.partition import manifest_key


@pytest.fixture
def streamed(tmp_path, sample_csv):
    """An index shaped the way the streaming indexer leaves one: documents
    split across partitions, a manifest each, offsets restarting at zero in
    every partition."""
    store = LocalStore(tmp_path)
    events = list(iter_events(sample_csv))
    partitions = 3
    for partition in range(partitions):
        mine = events[partition::partitions]
        key = f"segments/p{partition}/00000000000000000000-{len(mine) - 1:020d}.seg"
        index = build_index(mine)
        store.put(key, write_segment(index))
        meta = SegmentMeta(
            key=key,
            docs=len(mine),
            bytes=store.size(key),
            min_ts=min(e["ts"] for e in mine),
            max_ts=max(e["ts"] for e in mine),
            # Offsets are per partition, so every one of these starts at zero.
            first_offset=0,
            last_offset=len(mine) - 1,
        )
        write_manifest(store, Manifest().publish(meta), manifest_key(partition))
    return store, len(events)


def test_partitions_do_not_evict_each_other(streamed):
    """The trap in unioning these.

    `Manifest.publish` evicts by offset range, which is correct inside a
    partition and destroys the index across them: offsets are per partition,
    so every partition holds a segment covering offset 0, and each would evict
    the others. Union deduplicates by object key instead.
    """
    store, total = streamed
    manifests = [read_manifest(store, manifest_key(p)) for p in range(3)]
    assert all(m.segments[0].first_offset == 0 for m in manifests)

    merged = union_manifests(manifests)
    assert len(merged.segments) == 3
    assert merged.docs == total

    # What publishing would have done, for contrast.
    naive = Manifest()
    for manifest in manifests:
        for segment in manifest.segments:
            naive = naive.publish(segment)
    assert len(naive.segments) == 1, "publish across partitions collapses the index"


def test_the_coordinator_searches_every_partition(streamed, sample_csv):
    store, total = streamed
    whole = Coordinator(store, manifest_prefix="manifests/")
    assert whole.num_docs == total

    reference = build_index(iter_events(sample_csv))
    result = whole.search("samsung smartphone", top_k=100)
    assert result.total == len(reference.search_and("samsung smartphone"))


def test_it_finds_documents_a_single_partition_would_miss(streamed):
    """The failure this fixes: one partition holds a third of the answers."""
    store, _ = streamed
    whole = Coordinator(store, manifest_prefix="manifests/").search("samsung", top_k=100)
    one = Coordinator(store, manifest_key=manifest_key(0)).search("samsung", top_k=100)
    assert whole.total > one.total


def test_an_empty_prefix_is_an_empty_index_not_an_error(tmp_path):
    """A service can legitimately start before its first ingest."""
    coordinator = Coordinator(LocalStore(tmp_path), manifest_prefix="manifests/")
    assert coordinator.num_docs == 0
    assert coordinator.search("samsung").total == 0


def test_refresh_picks_up_a_new_partition(streamed):
    """Partitions appear as a consumer group grows, and a running coordinator
    must notice rather than serving whatever existed when it started."""
    store, total = streamed
    coordinator = Coordinator(store, manifest_prefix="manifests/")
    assert coordinator.num_docs == total

    index = build_index([{
        "event_id": "extra", "ts": 1_570_000_000, "session_id": "s", "user_id": "u",
        "event_type": "view", "device": None, "product_id": "p", "title": "a lamp",
        "category": "home.lamp", "brand": "ikea", "price": 9.0, "query": None,
    }])
    key = "segments/p9/00000000000000000000-00000000000000000000.seg"
    store.put(key, write_segment(index))
    write_manifest(
        store,
        Manifest().publish(SegmentMeta(key, 1, store.size(key), 1, 1, 0, 0)),
        manifest_key(9),
    )

    coordinator.refresh()
    assert coordinator.num_docs == total + 1
    assert coordinator.search("lamp", top_k=5).total == 1
