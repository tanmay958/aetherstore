"""The object store contract, checked against every implementation.

LocalStore stands in for S3 during development, so the contract it implements
has to be the one S3 actually offers: whole-object writes, inclusive byte
ranges, and a suffix read that works without knowing the object's size.
Anything the local implementation allows that S3 would not is a trap waiting
for the deploy.

Running both through identical tests is what makes "swap the store" a config
change rather than a leap of faith.
"""

import pytest
from fake_s3 import FakeS3Client

from aether.storage import LocalStore, open_store
from aether.storage.s3 import S3Store


@pytest.fixture(params=["local", "s3"])
def store(request, tmp_path):
    if request.param == "local":
        store = LocalStore(tmp_path)
    else:
        store = S3Store("bucket", client=FakeS3Client())
    store.put("obj", b"0123456789")
    return store


def test_round_trips_an_object(store):
    assert store.get_range("obj", 0, 10) == b"0123456789"
    assert store.size("obj") == 10
    assert store.exists("obj")


@pytest.mark.parametrize(
    "start, length, expected",
    [
        (0, 1, b"0"),
        (2, 3, b"234"),
        (0, 10, b"0123456789"),
        (9, 1, b"9"),
    ],
)
def test_reads_an_arbitrary_range(store, start, length, expected):
    """Length, not end offset. HTTP ranges are inclusive at both ends, so the
    conversion is where an off-by-one would hide."""
    assert store.get_range("obj", start, length) == expected


def test_reading_past_the_end_returns_what_exists(store):
    assert store.get_range("obj", 8, 100) == b"89"


def test_reading_entirely_past_the_end_returns_nothing(store):
    assert store.get_range("obj", 50, 10) == b""


def test_zero_length_read_returns_nothing(store):
    assert store.get_range("obj", 0, 0) == b""


def test_suffix_read_needs_no_knowledge_of_the_size(store):
    """This is `Range: bytes=-N`, and it is how a reader finds a segment's
    footer in a file it has never seen and whose length it does not know."""
    assert store.get_suffix("obj", 4) == b"6789"


def test_suffix_longer_than_the_object_returns_the_object(store):
    assert store.get_suffix("obj", 999) == b"0123456789"


def test_zero_length_suffix_returns_nothing(store):
    assert store.get_suffix("obj", 0) == b""


def test_put_replaces_wholly(store):
    """No append and no partial update, because object storage offers
    neither. That constraint is why segments are immutable."""
    store.put("obj", b"xy")
    assert store.get_range("obj", 0, 10) == b"xy"
    assert store.size("obj") == 2


def test_nested_keys_work(store):
    store.put("segments/p0/000-099.seg", b"data")
    assert store.exists("segments/p0/000-099.seg")
    assert store.get_range("segments/p0/000-099.seg", 0, 4) == b"data"


def test_missing_object_is_absent(store):
    assert not store.exists("nope")


def test_reading_a_missing_object_raises_the_same_error_everywhere(store):
    """LocalStore raises FileNotFoundError, so S3Store translates its client
    errors into one too. Callers cannot afford to care which store they have."""
    with pytest.raises(FileNotFoundError):
        store.get_range("nope", 0, 1)
    with pytest.raises(FileNotFoundError):
        store.size("nope")


# --------------------------------------------------------------------------
# LocalStore only
# --------------------------------------------------------------------------


def test_local_keys_cannot_escape_the_store_root(tmp_path):
    """Keys will eventually come from manifests written by other processes."""
    store = LocalStore(tmp_path)
    with pytest.raises(ValueError, match="escapes the store root"):
        store.get_range("../../etc/passwd", 0, 1)


# --------------------------------------------------------------------------
# choosing a store
# --------------------------------------------------------------------------


@pytest.mark.parametrize("uri", ["data/segments", "/tmp/x", "file:///tmp/x"])
def test_paths_open_a_local_store(uri):
    assert isinstance(open_store(uri), LocalStore)


def test_s3_uri_opens_an_s3_store(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    store = open_store("s3://aether/indexes/click")
    assert isinstance(store, S3Store)
    assert store.bucket == "aether"
    assert store.prefix == "indexes/click"


def test_s3_uri_without_a_prefix(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    assert open_store("s3://aether").prefix == ""


def test_rejects_a_bucketless_s3_uri():
    with pytest.raises(ValueError, match="names no bucket"):
        open_store("s3:///justapath")


def test_rejects_an_unknown_scheme():
    with pytest.raises(ValueError, match="unsupported storage scheme"):
        open_store("gopher://example.com/x")


# --------------------------------------------------------------------------
# delete
# --------------------------------------------------------------------------


def test_delete_removes_an_object(store):
    store.delete("obj")
    assert not store.exists("obj")


def test_deleting_something_absent_is_not_an_error(store):
    """A retried delete is the normal case, not the exception: compaction and
    orphan collection both re-run after failures."""
    store.delete("never-existed")
    store.delete("obj")
    store.delete("obj")
