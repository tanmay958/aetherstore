"""End to end against a real S3-compatible backend.

Skipped unless a backend is actually reachable, so the default suite stays
offline. What runs here is the thing a fake client cannot prove: that a real
implementation honours inclusive ranges, suffix ranges, and the S3 dialect
this code assumes, and that a segment written to it can be searched by byte
range afterwards.

MinIO:
    make up
    uv run pytest -m integration

Cloudflare R2:
    export R2_ACCOUNT_ID=...  AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
    export AETHER_TEST_R2_BUCKET=aether
    uv run pytest -m integration
"""

from __future__ import annotations

import os
import socket
import uuid

import pytest

from aether.data.rees46 import iter_events
from aether.index.memory import build_index
from aether.index.segment import SegmentReader, write_segment
from aether.storage import CountingStore
from aether.storage.s3 import S3Store

pytestmark = pytest.mark.integration

MINIO_ENDPOINT = os.getenv("AETHER_TEST_MINIO_ENDPOINT", "http://localhost:9000")
MINIO_BUCKET = os.getenv("AETHER_TEST_MINIO_BUCKET", "aether")
R2_BUCKET = os.getenv("AETHER_TEST_R2_BUCKET")


def _reachable(url: str) -> bool:
    host_port = url.split("://", 1)[-1]
    host, _, port = host_port.partition(":")
    try:
        socket.create_connection((host, int(port or 80)), timeout=0.4).close()
    except OSError:
        return False
    return True


def _minio() -> S3Store:
    return S3Store(
        MINIO_BUCKET,
        prefix=f"itest/{uuid.uuid4().hex[:8]}",
        endpoint_url=MINIO_ENDPOINT,
    )


def _r2() -> S3Store:
    from aether.storage.factory import r2_credentials, r2_endpoint

    access, secret = r2_credentials()
    return S3Store(
        R2_BUCKET,
        prefix=f"itest/{uuid.uuid4().hex[:8]}",
        endpoint_url=r2_endpoint(),
        access_key=access,
        secret_key=secret,
    )


BACKENDS = [
    pytest.param(
        _minio,
        id="minio",
        marks=pytest.mark.skipif(
            not _reachable(MINIO_ENDPOINT), reason="MinIO not running; try `make up`"
        ),
    ),
    pytest.param(
        _r2,
        id="r2",
        marks=pytest.mark.skipif(
            not (R2_BUCKET and os.getenv("R2_ACCOUNT_ID")),
            reason="set R2_ACCOUNT_ID and AETHER_TEST_R2_BUCKET to test R2",
        ),
    ),
]


@pytest.fixture(params=BACKENDS)
def store(request):
    store = request.param()
    written: list[str] = []

    original_put = store.put

    def tracking_put(key: str, data: bytes) -> None:
        written.append(key)
        original_put(key, data)

    store.put = tracking_put  # type: ignore[method-assign]
    yield store
    for key in written:
        store.delete(key)


def test_round_trips_an_object(store):
    store.put("probe", bytes(range(256)))
    assert store.size("probe") == 256
    assert store.exists("probe")


def test_ranges_are_inclusive_on_a_real_backend(store):
    """The claim the fake client can only assume."""
    payload = bytes(range(256))
    store.put("probe", payload)
    assert store.get_range("probe", 0, 64) == payload[:64]
    assert store.get_range("probe", 100, 16) == payload[100:116]


def test_suffix_range_without_knowing_the_size(store):
    payload = bytes(range(256))
    store.put("probe", payload)
    assert store.get_suffix("probe", 116) == payload[-116:]


def test_range_past_the_end_returns_empty(store):
    store.put("probe", b"short")
    assert store.get_range("probe", 9999, 10) == b""


def test_missing_object_raises_file_not_found(store):
    with pytest.raises(FileNotFoundError):
        store.get_range("definitely-absent", 0, 1)


def test_delete_is_idempotent(store):
    store.put("probe", b"x")
    store.delete("probe")
    store.delete("probe")
    assert not store.exists("probe")


def test_a_segment_written_here_is_searchable_by_byte_range(store, sample_csv):
    """The whole point of step 7: identical results, over the network, with
    no change to the reader."""
    index = build_index(iter_events(sample_csv))
    store.put("0.seg", write_segment(index))

    counting = CountingStore(store)
    segment = SegmentReader(counting, "0.seg")

    assert segment.num_docs == index.num_docs
    assert segment.search_and("samsung smartphone") == index.search_and(
        "samsung smartphone"
    )

    expected = index.search("samsung smartphone", top_k=10)
    actual = segment.search("samsung smartphone", top_k=10)
    assert [h.doc_id for h in actual.hits] == [h.doc_id for h in expected.hits]
    assert [h.score for h in actual.hits] == [h.score for h in expected.hits]

    # The number that matters remotely: a whole query in a handful of
    # round trips, having read a fraction of the object.
    assert counting.stats.requests <= 8
    assert counting.stats.bytes_read < store.size("0.seg")
