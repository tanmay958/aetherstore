"""Tests specific to the S3-compatible store.

The generic contract lives in test_storage.py and runs against this too. What
is checked here is the part that has no equivalent locally: the exact HTTP
requests produced, and the configuration differences that decide whether R2
accepts them at all.
"""

import pytest
from fake_s3 import FakeS3Client

from aether.storage.s3 import R2_HOST_SUFFIX, S3Store, _build_client, default_region


# Client construction resolves credentials, so without this the tests would
# pick up whatever profile the developer happens to have configured and fail
# differently on every machine.
@pytest.fixture(autouse=True)
def isolated_aws_environment(monkeypatch):
    for name in (
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AETHER_S3_ENDPOINT",
        "R2_ACCOUNT_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


@pytest.fixture
def client() -> FakeS3Client:
    return FakeS3Client()


@pytest.fixture
def store(client) -> S3Store:
    store = S3Store("bucket", client=client)
    store.put("obj", bytes(range(256)))
    return store


# --------------------------------------------------------------------------
# range headers, where the off-by-one lives
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "start, length, header",
    [
        (0, 1, "bytes=0-0"),
        (0, 64, "bytes=0-63"),
        (100, 16, "bytes=100-115"),
        (255, 1, "bytes=255-255"),
    ],
)
def test_range_headers_are_inclusive_at_both_ends(store, client, start, length, header):
    """HTTP ranges include the last byte, so N bytes from offset S ends at
    S + N - 1. Off by one here returns an extra byte per read, which a varint
    decoder consumes as the start of a value that does not exist, and the
    corruption looks nothing like an off-by-one."""
    store.get_range("obj", start, length)
    assert client.range_headers[-1] == header


def test_range_returns_exactly_the_bytes_asked_for(store):
    assert store.get_range("obj", 100, 16) == bytes(range(100, 116))


def test_suffix_header_has_no_start_offset(store, client):
    """`bytes=-116` is what makes the footer findable without knowing the
    object's size."""
    store.get_suffix("obj", 116)
    assert client.range_headers[-1] == "bytes=-116"


def test_zero_length_reads_never_reach_the_network(store, client):
    """`bytes=-0` is malformed, and a zero-length range is pointless, so
    neither is sent."""
    before = len(client.range_headers)
    assert store.get_range("obj", 0, 0) == b""
    assert store.get_suffix("obj", 0) == b""
    assert len(client.range_headers) == before


# --------------------------------------------------------------------------
# keys and prefixes
# --------------------------------------------------------------------------


def test_prefix_is_prepended_to_every_key(client):
    store = S3Store("bucket", prefix="indexes/click", client=client)
    store.put("p0/0.seg", b"data")
    assert ("bucket", "indexes/click/p0/0.seg") in client.objects


def test_prefix_slashes_are_normalized(client):
    store = S3Store("bucket", prefix="/indexes/click/", client=client)
    store.put("0.seg", b"data")
    assert ("bucket", "indexes/click/0.seg") in client.objects


def test_no_prefix_leaves_keys_alone(client):
    S3Store("bucket", client=client).put("0.seg", b"data")
    assert ("bucket", "0.seg") in client.objects


# --------------------------------------------------------------------------
# errors translated to match LocalStore
# --------------------------------------------------------------------------


def test_missing_object_raises_file_not_found(store):
    """Callers cannot afford to know which store they are talking to, so
    client errors become the same exception LocalStore raises."""
    with pytest.raises(FileNotFoundError, match="bucket/absent"):
        store.get_range("absent", 0, 1)


def test_missing_object_size_raises_file_not_found(store):
    with pytest.raises(FileNotFoundError):
        store.size("absent")


def test_exists_is_false_rather_than_raising(store):
    assert store.exists("absent") is False


def test_range_past_the_end_returns_empty_rather_than_raising(store):
    """S3 answers 416 InvalidRange; LocalStore returns nothing. The interface
    promises the latter."""
    assert store.get_range("obj", 9999, 10) == b""


# --------------------------------------------------------------------------
# backend-specific configuration
# --------------------------------------------------------------------------


def test_r2_signs_with_the_auto_region():
    """R2 rejects any other region outright."""
    assert default_region(f"https://acc123.{R2_HOST_SUFFIX}") == "auto"


def test_other_custom_endpoints_get_a_concrete_region():
    """MinIO ignores the region but SigV4 still needs one to sign with."""
    assert default_region("http://localhost:9000") == "us-east-1"


def test_aws_resolves_its_own_region():
    assert default_region(None) is None


def test_custom_endpoints_disable_upload_checksums():
    """boto3 1.36 began attaching CRC32 checksums by default, which R2
    rejects with "Header 'x-amz-checksum-algorithm' with value 'CRC32' not
    implemented"."""
    config = _build_client("http://localhost:9000", "us-east-1", 8).meta.config
    assert config.request_checksum_calculation == "when_required"
    assert config.response_checksum_validation == "when_required"


def test_aws_keeps_its_checksums():
    """The workaround is a compatibility shim, not an improvement, so real S3
    keeps its upload integrity checks."""
    config = _build_client(None, "us-east-1", 8).meta.config
    assert config.request_checksum_calculation != "when_required"


def test_connection_pool_is_wide_enough_for_fan_out():
    """The query planner will fan out across segments concurrently, and a
    narrow pool would make connection reuse the bottleneck instead of the
    network."""
    assert _build_client(None, "us-east-1", 64).meta.config.max_pool_connections == 64
