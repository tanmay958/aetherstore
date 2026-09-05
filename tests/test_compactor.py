"""Tests for compaction.

Two claims to defend. A compacted index must answer exactly what it answered
before, because merging is a storage decision and relevance is not. And a
crash at any point must leave the index correct, because compaction rewrites
the live set and a half-finished merge is the worst possible state to be
stuck in.
"""

import pytest
from failing_store import CrashAfter

from aether.data.rees46 import iter_events
from aether.index.compactor import (
    SIZE_TIER_RATIO,
    CompactionPlan,
    compact,
    merge_group,
    merged_key,
    plan_compaction,
)
from aether.index.coordinator import Coordinator
from aether.index.ingest import ingest
from aether.index.manifest import Manifest, SegmentMeta, read_manifest
from aether.storage import LocalStore


def meta(key: str, *, first: int, last: int, size: int = 1000, docs: int = 10):
    return SegmentMeta(key, docs, size, 0, 0, first_offset=first, last_offset=last)


def uniform(count: int, size: int = 1000) -> Manifest:
    return Manifest().with_segments(
        [meta(f"s{i}.seg", first=i * 10, last=i * 10 + 9, size=size) for i in range(count)]
    )


@pytest.fixture
def indexed(tmp_path, sample_csv):
    """The fixture split into nine segments of three documents."""
    store = LocalStore(tmp_path)
    ingest(iter_events(sample_csv), store, docs_per_segment=3)
    return store


# --------------------------------------------------------------------------
# planning, which needs no storage at all
# --------------------------------------------------------------------------


def test_nothing_to_do_below_the_merge_factor():
    assert not plan_compaction(uniform(5), merge_factor=10)


def test_a_full_run_becomes_one_merge():
    plan = plan_compaction(uniform(10), merge_factor=10)
    assert plan.segments_in == 10
    assert plan.segments_out == 1


def test_leftovers_are_left_alone():
    """Twenty-five segments at factor ten is two merges and five untouched."""
    plan = plan_compaction(uniform(25), merge_factor=10)
    assert plan.segments_out == 2
    assert plan.segments_in == 20


def test_groups_are_adjacent_in_offset_order():
    """Merging non-adjacent segments would produce a range with a hole in it,
    and a later replay landing inside that hole would evict a merged segment
    holding thousands of unrelated documents."""
    plan = plan_compaction(uniform(20), merge_factor=10)
    for group in plan.groups:
        offsets = [segment.first_offset for segment in group]
        assert offsets == sorted(offsets)
        for earlier, later in zip(group, group[1:]):
            assert earlier.last_offset < later.first_offset


def test_planning_does_not_depend_on_manifest_order():
    forward = uniform(10)
    shuffled = Manifest().with_segments(list(reversed(forward.segments)))
    assert [s.key for s in plan_compaction(forward, merge_factor=10).groups[0]] == [
        s.key for s in plan_compaction(shuffled, merge_factor=10).groups[0]
    ]


def test_a_much_larger_segment_starts_a_new_tier():
    """Merging a big segment with small ones rewrites all of it to save a
    single request."""
    segments = [meta(f"s{i}.seg", first=i * 10, last=i * 10 + 9, size=1000) for i in range(5)]
    segments.append(meta("big.seg", first=50, last=59, size=int(1000 * SIZE_TIER_RATIO * 4)))
    segments += [meta(f"t{i}.seg", first=60 + i * 10, last=69 + i * 10, size=1000) for i in range(5)]

    plan = plan_compaction(Manifest().with_segments(segments), merge_factor=5)
    for group in plan.groups:
        assert "big.seg" not in {s.key for s in group}


def test_a_merge_stops_at_the_size_ceiling():
    plan = plan_compaction(uniform(20, size=1000), merge_factor=2, max_merged_bytes=3500)
    for group in plan.groups:
        assert sum(segment.bytes for segment in group) <= 3500


def test_rejects_a_nonsense_merge_factor():
    with pytest.raises(ValueError, match="at least 2"):
        plan_compaction(uniform(10), merge_factor=1)


def test_an_empty_plan_is_falsey():
    assert not CompactionPlan()


# --------------------------------------------------------------------------
# naming
# --------------------------------------------------------------------------


def test_merged_key_covers_the_whole_range():
    """Derived from the inputs, not from a counter or a clock, so re-running
    an identical merge overwrites an identical object."""
    group = [meta("segments/a.seg", first=0, last=99), meta("segments/b.seg", first=100, last=199)]
    assert merged_key(group) == "segments/" + f"{0:020d}-{199:020d}.seg"


def test_merged_key_is_stable():
    group = [meta("segments/a.seg", first=0, last=99), meta("segments/b.seg", first=100, last=199)]
    assert merged_key(group) == merged_key(list(group))


# --------------------------------------------------------------------------
# the merge itself
# --------------------------------------------------------------------------


def test_compaction_reduces_the_segment_count(indexed):
    """Nine segments at a merge factor of three is three merges, not one.
    Work per run is capped so repeated passes converge, rather than a single
    pass rewriting the whole index."""
    before = read_manifest(indexed)
    compact(indexed, merge_factor=3)
    after = read_manifest(indexed)

    assert len(before.segments) == 9
    assert len(after.segments) == 3
    assert after.docs == before.docs


def test_every_document_survives(indexed, sample_csv):
    compact(indexed, merge_factor=3)

    from aether.index.segment import SegmentReader

    live = sorted(read_manifest(indexed).segments, key=lambda s: s.first_offset)
    seen = []
    for segment in live:
        reader = SegmentReader(indexed, segment.key)
        seen.extend(reader.document(i)["event_id"] for i in range(reader.num_docs))

    assert seen == [event["event_id"] for event in iter_events(sample_csv)]


def test_queries_are_unchanged_by_compaction(indexed):
    """The claim that matters. How the data was chopped up is a storage
    decision, and relevance is not."""
    before = Coordinator(indexed)
    baseline = {
        query: before.search(query, top_k=10, global_stats=True)
        for query in ("samsung smartphone", "electronics", "bosch", "view purchase")
    }
    ids_before = {
        query: [before.document(hit)["event_id"] for hit in result.hits]
        for query, result in baseline.items()
    }

    compact(indexed, merge_factor=3)

    after = Coordinator(indexed)
    for query, expected in baseline.items():
        actual = after.search(query, top_k=10, global_stats=True)
        assert actual.total == expected.total
        assert [after.document(hit)["event_id"] for hit in actual.hits] == ids_before[query]
        assert [round(hit.score, 9) for hit in actual.hits] == [
            round(hit.score, 9) for hit in expected.hits
        ]


def test_sources_are_not_deleted(indexed):
    """A coordinator that read the manifest a moment ago is still reading
    them. Retiring them is the collector's job, after a grace period."""
    result = compact(indexed, merge_factor=3)
    assert result.retired
    assert all(indexed.exists(segment.key) for segment in result.retired)


def test_sources_are_unlisted_from_the_manifest(indexed):
    result = compact(indexed, merge_factor=3)
    live = {segment.key for segment in read_manifest(indexed).segments}
    assert not live & {segment.key for segment in result.retired}


def test_repeated_compaction_converges_and_loses_nothing(indexed):
    """Not idempotent, and should not be: a second pass merges what the first
    left. What must hold is that documents are neither lost nor duplicated,
    and that it eventually settles."""
    counts = [len(read_manifest(indexed).segments)]
    for _ in range(4):
        compact(indexed, merge_factor=3)
        live = read_manifest(indexed)
        counts.append(len(live.segments))
        assert live.docs == 27

    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == counts[-2]  # settled


def test_merging_the_same_group_twice_is_idempotent(indexed):
    """The narrower property that does hold, and the one crash recovery needs:
    redoing an interrupted merge rewrites the same object under the same key
    rather than adding a second copy."""
    group = list(read_manifest(indexed).segments[:3])

    first = merge_group(indexed, group, "manifest.json")
    bytes_first = indexed.get_range(first.key, 0, first.bytes)
    second = merge_group(indexed, group, "manifest.json")

    assert first.key == second.key
    assert indexed.get_range(second.key, 0, second.bytes) == bytes_first
    live = [s.key for s in read_manifest(indexed).segments]
    assert live.count(first.key) == 1


def test_a_second_pass_merges_what_the_first_left(tmp_path, sample_csv):
    store = LocalStore(tmp_path)
    ingest(iter_events(sample_csv), store, docs_per_segment=3)

    compact(store, merge_factor=2)
    first = len(read_manifest(store).segments)
    compact(store, merge_factor=2)
    second = len(read_manifest(store).segments)

    assert 9 > first > second


def test_a_segment_published_during_a_merge_is_not_lost(indexed):
    """The manifest is re-read at commit time rather than trusting a copy
    taken before the merge began."""
    manifest = read_manifest(indexed)
    group = list(manifest.segments[:3])

    # Something else publishes while the merge is in flight.
    from aether.index.manifest import write_manifest

    newcomer = meta("segments/late.seg", first=9000, last=9009)
    write_manifest(indexed, manifest.publish(newcomer))

    merge_group(indexed, group, "manifest.json")

    live = {segment.key for segment in read_manifest(indexed).segments}
    assert "segments/late.seg" in live


# --------------------------------------------------------------------------
# crashes
# --------------------------------------------------------------------------


def test_crash_before_the_merged_segment_changes_nothing(indexed):
    before = read_manifest(indexed)
    group = list(before.segments[:3])

    with pytest.raises(CrashAfter.Crash):
        merge_group(CrashAfter(indexed, writes=0), group, "manifest.json")

    assert read_manifest(indexed).segments == before.segments
    assert all(indexed.exists(segment.key) for segment in group)


def test_crash_after_the_segment_but_before_the_manifest(indexed):
    """The merged object exists and nothing points at it, so no query can see
    it. The sources are still live and the index is still correct."""
    before = read_manifest(indexed)
    group = list(before.segments[:3])

    with pytest.raises(CrashAfter.Crash):
        merge_group(CrashAfter(indexed, writes=1), group, "manifest.json")

    assert indexed.exists(merged_key(group))
    assert read_manifest(indexed).segments == before.segments

    coordinator = Coordinator(indexed)
    assert coordinator.search("samsung smartphone", top_k=10).total > 0


def test_a_crashed_merge_can_simply_be_re_run(indexed):
    group = list(read_manifest(indexed).segments[:3])
    with pytest.raises(CrashAfter.Crash):
        merge_group(CrashAfter(indexed, writes=1), group, "manifest.json")

    merged = merge_group(indexed, group, "manifest.json")

    live = {segment.key for segment in read_manifest(indexed).segments}
    assert merged.key in live
    assert not live & {segment.key for segment in group}
