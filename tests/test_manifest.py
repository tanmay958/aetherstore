"""Tests for the manifest.

The manifest is the commit point: a segment is invisible until named here, and
visible the instant it is. Everything below protects that property or the
pruning metadata that makes a query cheap.
"""

import pytest

from aether.index.manifest import Manifest, SegmentMeta, read_manifest, write_manifest
from aether.storage import LocalStore

A = SegmentMeta("segments/a.seg", docs=10, bytes=2000, min_ts=100, max_ts=200)
B = SegmentMeta("segments/b.seg", docs=15, bytes=3000, min_ts=300, max_ts=400)


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


def test_absent_manifest_is_an_empty_index_not_an_error(store):
    """A bucket with nothing in it is a valid starting state."""
    assert read_manifest(store).segments == ()


def test_round_trips(store):
    write_manifest(store, Manifest().with_segments([A, B]))
    restored = read_manifest(store)
    assert restored.segments == (A, B)
    assert restored.docs == 25
    assert restored.bytes == 5000


def test_generation_advances_on_every_publish():
    """Not used for coordination yet. It exists because a rebalancing consumer
    group can briefly leave two writers believing they own the same partition,
    and a monotonic number is what lets a zombie be detected."""
    first = Manifest().with_segments([A])
    assert first.generation == 1
    assert first.with_segments([A, B]).generation == 2


def test_replacing_the_segment_set_is_one_write(store):
    """How compaction swaps ten small segments for one large one."""
    write_manifest(store, Manifest().with_segments([A, B]))
    merged = SegmentMeta("segments/merged.seg", 25, 4500, 100, 400)
    write_manifest(store, read_manifest(store).with_segments([merged]))

    live = read_manifest(store)
    assert [s.key for s in live.segments] == ["segments/merged.seg"]
    assert live.generation == 2


def test_rejects_an_unsupported_version(store):
    store.put("manifest.json", b'{"version":99,"segments":[]}')
    with pytest.raises(ValueError, match="version 99 is not supported"):
        read_manifest(store)


def test_rejects_garbage(store):
    store.put("manifest.json", b"not json at all")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_manifest(store)


# --------------------------------------------------------------------------
# pruning
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "start, end, expected",
    [
        (None, None, True),      # no filter
        (150, 250, True),        # overlaps the middle
        (100, 200, True),        # exactly the span
        (200, 500, True),        # touches the last instant
        (0, 100, True),          # touches the first instant
        (201, 500, False),       # entirely after
        (0, 99, False),          # entirely before
        (300, None, False),      # open-ended, after
        (None, 50, False),       # open-ended, before
    ],
)
def test_time_overlap_decides_pruning(start, end, expected):
    """The cheapest optimization in the engine: whole segments discarded on
    arithmetic over data already in memory, for zero requests."""
    assert A.overlaps(start, end) is expected
