"""Tests for the object store interface.

LocalStore stands in for S3 during development, so the contract it implements
has to be the one S3 actually offers: whole-object writes, range reads, and a
suffix read that works without knowing the object's size. Anything the local
implementation allows that S3 would not is a trap for later.
"""

import pytest

from aether.storage import LocalStore


@pytest.fixture
def store(tmp_path):
    store = LocalStore(tmp_path)
    store.put("obj", b"0123456789")
    return store


def test_round_trips_an_object(store):
    assert store.get_range("obj", 0, 10) == b"0123456789"
    assert store.size("obj") == 10
    assert store.exists("obj")


def test_reads_an_arbitrary_range(store):
    assert store.get_range("obj", 2, 3) == b"234"
    assert store.get_range("obj", 0, 1) == b"0"


def test_reading_past_the_end_returns_what_exists(store):
    """Matching HTTP range behaviour, which truncates rather than erroring."""
    assert store.get_range("obj", 8, 100) == b"89"
    assert store.get_range("obj", 50, 10) == b""


def test_suffix_read_needs_no_knowledge_of_the_size(store):
    """This is what "Range: bytes=-N" does, and it is how a reader finds a
    segment's footer in a file it has never seen."""
    assert store.get_suffix("obj", 4) == b"6789"


def test_suffix_longer_than_the_object_returns_the_object(store):
    assert store.get_suffix("obj", 999) == b"0123456789"


def test_put_replaces_wholly(store):
    """There is no append and no partial update, because object storage has
    neither. That constraint is why segments are immutable."""
    store.put("obj", b"xy")
    assert store.get_range("obj", 0, 10) == b"xy"
    assert store.size("obj") == 2


def test_creates_intermediate_directories(store):
    store.put("segments/p0/000-099.seg", b"data")
    assert store.exists("segments/p0/000-099.seg")


def test_missing_object_is_absent(store):
    assert not store.exists("nope")
    with pytest.raises(FileNotFoundError):
        store.get_range("nope", 0, 1)


def test_keys_cannot_escape_the_store_root(store):
    """Keys will eventually come from manifests written by other processes."""
    with pytest.raises(ValueError, match="escapes the store root"):
        store.get_range("../../etc/passwd", 0, 1)
