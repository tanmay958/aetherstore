"""Tests for orphan collection.

Every other component in this engine fails safe: a bad segment is invisible, a
crashed flush is replayed, a wrong score is embarrassing. Deleting an object
is the one operation with no undo, so these tests are mostly about what the
collector must refuse to touch rather than what it removes.
"""

import pytest

from aether.data.rees46 import iter_events
from aether.index.compactor import compact
from aether.index.coordinator import Coordinator
from aether.index.gc import DEFAULT_GRACE_SECONDS, collect, find_manifests
from aether.index.ingest import ingest
from aether.index.manifest import Manifest, SegmentMeta, read_manifest, write_manifest
from aether.storage import LocalStore

HOUR = 3600.0


@pytest.fixture
def indexed(tmp_path, sample_csv):
    store = LocalStore(tmp_path)
    ingest(iter_events(sample_csv), store, docs_per_segment=3)
    return store


def aged(store, key: str, seconds: float) -> None:
    """Backdate an object, so a grace period can be tested without waiting."""
    import os

    path = store._path(key)
    stamp = path.stat().st_mtime - seconds
    os.utime(path, (stamp, stamp))


# --------------------------------------------------------------------------
# what it must never delete
# --------------------------------------------------------------------------


def test_never_deletes_a_referenced_segment(indexed):
    """However old it is."""
    for segment in read_manifest(indexed).segments:
        aged(indexed, segment.key, 10 * HOUR)

    result = collect(indexed, delete=True)

    assert result.deleted == []
    assert all(indexed.exists(s.key) for s in read_manifest(indexed).segments)


def test_never_deletes_a_recent_orphan(indexed):
    """An unreferenced object may be a segment whose manifest has not landed
    yet, which is one line of the flush away from being live."""
    indexed.put("segments/just-written.seg", b"x" * 100)

    result = collect(indexed, delete=True)

    assert indexed.exists("segments/just-written.seg")
    assert [orphan.key for orphan in result.spared] == ["segments/just-written.seg"]


def test_never_deletes_a_manifest(indexed):
    aged(indexed, "manifest.json", 10 * HOUR)
    collect(indexed, delete=True)
    assert indexed.exists("manifest.json")


def test_a_dry_run_deletes_nothing(indexed):
    indexed.put("segments/orphan.seg", b"x" * 100)
    aged(indexed, "segments/orphan.seg", 10 * HOUR)

    result = collect(indexed)  # delete defaults to False

    assert result.dry_run is True
    assert [orphan.key for orphan in result.deleted] == ["segments/orphan.seg"]
    assert indexed.exists("segments/orphan.seg")


def test_missing_a_manifest_would_be_catastrophic_so_all_are_found(tmp_path, sample_csv):
    """The streaming indexer writes one manifest per Kafka partition. Reading
    only some of them would make every segment in the others look unreferenced
    and delete a live index."""
    store = LocalStore(tmp_path)
    for partition in range(3):
        segment = SegmentMeta(
            f"segments/p{partition}/0.seg", 1, 10, 0, 0, first_offset=0, last_offset=0
        )
        store.put(segment.key, b"data")
        write_manifest(
            store,
            Manifest().with_segments([segment]),
            f"manifests/p{partition}/current.json",
        )
    for partition in range(3):
        aged(store, f"segments/p{partition}/0.seg", 10 * HOUR)

    assert len(find_manifests(store)) == 3
    result = collect(store, delete=True)
    assert result.deleted == []
    # Three segments plus the three manifests: manifests are protected too.
    assert result.protected == 6


# --------------------------------------------------------------------------
# what it does delete
# --------------------------------------------------------------------------


def test_deletes_an_old_unreferenced_object(indexed):
    indexed.put("segments/orphan.seg", b"x" * 500)
    aged(indexed, "segments/orphan.seg", 10 * HOUR)

    result = collect(indexed, delete=True)

    assert [orphan.key for orphan in result.deleted] == ["segments/orphan.seg"]
    assert result.bytes_deleted == 500
    assert not indexed.exists("segments/orphan.seg")


def test_cleans_up_after_compaction(indexed):
    """The case this exists for: every merge retires its inputs."""
    result = compact(indexed, merge_factor=3)
    assert result.retired

    for segment in result.retired:
        aged(indexed, segment.key, 10 * HOUR)

    collected = collect(indexed, delete=True)

    assert {orphan.key for orphan in collected.deleted} == {
        segment.key for segment in result.retired
    }
    assert all(indexed.exists(s.key) for s in read_manifest(indexed).segments)


def test_the_index_still_answers_after_collection(indexed):
    before = Coordinator(indexed).search("samsung smartphone", top_k=10, global_stats=True)

    compact(indexed, merge_factor=3)
    for key in list(indexed.list_keys("segments/")):
        aged(indexed, key, 10 * HOUR)
    collect(indexed, delete=True)

    after = Coordinator(indexed).search("samsung smartphone", top_k=10, global_stats=True)
    assert after.total == before.total
    assert [round(h.score, 9) for h in after.hits] == [
        round(h.score, 9) for h in before.hits
    ]


def test_an_in_flight_reader_survives_a_collection(indexed):
    """A coordinator that opened segments before compaction keeps working,
    because the collector leaves recent objects alone. This is the scenario
    the grace period exists for."""
    reader = Coordinator(indexed)
    reader.search("samsung smartphone", top_k=10)  # opens every segment

    compact(indexed, merge_factor=3)
    collect(indexed, delete=True)  # retired sources are seconds old, so spared

    assert reader.search("samsung smartphone", top_k=10).total > 0


# --------------------------------------------------------------------------
# the grace period
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "age_hours, grace_hours, expect_deleted",
    [(10, 1, True), (0.5, 1, False), (2, 1, True), (1.5, 2, False)],
)
def test_the_grace_period_decides(indexed, age_hours, grace_hours, expect_deleted):
    indexed.put("segments/orphan.seg", b"x")
    aged(indexed, "segments/orphan.seg", age_hours * HOUR)

    result = collect(indexed, grace_seconds=grace_hours * HOUR, delete=True)

    assert bool(result.deleted) is expect_deleted
    assert indexed.exists("segments/orphan.seg") is not expect_deleted


def test_the_default_grace_is_generous():
    """Nothing measures this. It is deliberately far longer than either an
    in-flight query or a half-published flush needs."""
    assert DEFAULT_GRACE_SECONDS >= HOUR


def test_rejects_a_negative_grace(indexed):
    with pytest.raises(ValueError, match="must not be negative"):
        collect(indexed, grace_seconds=-1)


def test_reports_what_it_scanned(indexed):
    result = collect(indexed)
    assert result.scanned == len(list(indexed.list_keys()))
    assert result.protected == len(read_manifest(indexed).segments) + 1  # + manifest


def test_an_empty_store_is_not_an_error(tmp_path):
    result = collect(LocalStore(tmp_path), delete=True)
    assert result.scanned == 0
    assert result.deleted == []
