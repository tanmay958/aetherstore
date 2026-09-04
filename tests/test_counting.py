"""Tests for storage cost accounting.

The request count is the claim the whole segment format rests on, so the thing
that measures it needs to be right. In particular `measure()` has to isolate a
stretch of work: an early version of the search CLI aliased the live stats
object instead of snapshotting it, and every stage reported the running total.
"""

import pytest

from aether.storage import CountingStore, LocalStore, ReadStats


@pytest.fixture
def store(tmp_path):
    inner = LocalStore(tmp_path)
    inner.put("obj", b"0123456789")
    return CountingStore(inner)


def test_counts_requests_and_bytes(store):
    store.get_range("obj", 0, 4)
    store.get_range("obj", 4, 2)
    assert store.stats.requests == 2
    assert store.stats.bytes_read == 6


def test_counts_a_suffix_read(store):
    store.get_suffix("obj", 3)
    assert store.stats.requests == 1
    assert store.stats.bytes_read == 3


def test_counts_bytes_returned_not_bytes_asked_for(store):
    """A range running past the end returns less than requested, and the cost
    that matters is what actually moved."""
    store.get_range("obj", 8, 100)
    assert store.stats.bytes_read == 2


def test_writes_are_not_counted(store):
    """Writes cost too, but they happen once per segment during indexing
    rather than on every query, so they are not what the query path is being
    judged on."""
    store.put("other", b"data")
    assert store.stats.requests == 0


def test_metadata_calls_are_not_counted(store):
    store.size("obj")
    store.exists("obj")
    assert store.stats.requests == 0


def test_reset(store):
    store.get_range("obj", 0, 4)
    store.reset()
    assert store.stats.requests == 0
    assert store.stats.bytes_read == 0


def test_measure_isolates_one_stretch_of_work(store):
    store.get_range("obj", 0, 4)  # before

    with store.measure() as cost:
        store.get_range("obj", 0, 2)

    store.get_range("obj", 0, 5)  # after

    assert cost.requests == 1
    assert cost.bytes_read == 2


def test_measure_does_not_alias_the_live_stats(store):
    """The bug this exists to prevent: a stage's cost must stop accumulating
    once the stage is over."""
    with store.measure() as cost:
        store.get_range("obj", 0, 2)
    store.get_range("obj", 0, 9)

    assert cost.requests == 1
    assert cost.bytes_read == 2


def test_measure_reports_zero_when_nothing_was_read(store):
    with store.measure() as cost:
        pass
    assert cost.requests == 0


def test_measure_still_reports_when_the_body_raises(store):
    with pytest.raises(RuntimeError):
        with store.measure() as cost:
            store.get_range("obj", 0, 2)
            raise RuntimeError("boom")
    assert cost.requests == 1


def test_tracing_records_each_read(tmp_path):
    inner = LocalStore(tmp_path)
    inner.put("obj", b"0123456789")
    store = CountingStore(inner, trace=True)

    store.get_range("obj", 2, 3)
    store.get_suffix("obj", 4)

    assert [(r.key, r.start, r.returned) for r in store.reads] == [
        ("obj", 2, 3),
        ("obj", None, 4),  # a suffix read has no known start offset
    ]


def test_tracing_is_off_by_default(store):
    store.get_range("obj", 0, 4)
    assert store.reads == []


def test_reads_still_return_the_data(store):
    assert store.get_range("obj", 2, 3) == b"234"
    assert store.get_suffix("obj", 2) == b"89"


@pytest.mark.parametrize(
    "stats, expected",
    [
        (ReadStats(1, 64), "1 request, 64 B"),
        (ReadStats(3, 2048), "3 requests, 2.0 KB"),
        (ReadStats(0, 0), "0 requests, 0 B"),
        (ReadStats(2, 3_145_728), "2 requests, 3.00 MB"),
    ],
)
def test_readable_summary(stats, expected):
    assert str(stats) == expected
