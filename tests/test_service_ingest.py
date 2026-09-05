"""Indexing events posted over HTTP, and streaming them.

The property that matters is at the top: an event posted and then searched
for must be found. Everything the engine does between those two calls, buffer,
seal a segment, publish a manifest, refresh the reader, is invisible to
whoever pressed the button, and any of it being skipped shows up here.
"""

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from aether.data.rees46 import iter_events  # noqa: E402
from aether.index.ingest import ingest  # noqa: E402
from aether.service.app import create_app  # noqa: E402
from aether.service.feed import Feed, write_feed  # noqa: E402
from aether.service.state import ServiceState  # noqa: E402
from aether.storage import LocalStore  # noqa: E402

BASE_TS = 1_570_000_000


@pytest.fixture
def store(tmp_path, sample_csv):
    """A streamed-shaped index: per-partition manifests, room to add more.

    The batch ingester writes a single `manifest.json`, which a service
    configured with `manifest_prefix` will never look at. Republishing it as
    a partition manifest is what makes this fixture hold documents the tests
    can actually see; without it every test here ran against an empty index
    and several passed for the wrong reason.
    """
    from aether.index.manifest import read_manifest, write_manifest
    from aether.stream.partition import manifest_key

    store = LocalStore(tmp_path / "live")
    ingest(iter_events(sample_csv), store, docs_per_segment=8)
    write_manifest(store, read_manifest(store), manifest_key(0))
    store.delete("manifest.json")
    return store


@pytest.fixture
def writer(store):
    state = ServiceState(
        store, manifest_prefix="manifests/", model_uri=None,
        refresh_seconds=0.0001, writable=True,
    )
    state.load()
    return TestClient(create_app(state)), state


def an_event(title="Bespoke Widget XR9", **extra):
    return {
        "ts": BASE_TS,
        "event_type": "view",
        "session_id": "http-1",
        "product_id": "custom-1",
        "title": title,
        "brand": "acme",
        "category": "tools.widget",
        "price": 42.5,
        **extra,
    }


# --------------------------------------------------------------------------
# the property everything else exists for
# --------------------------------------------------------------------------


def test_a_posted_event_is_searchable_immediately(writer):
    """Post, then search, and find it. No sleeping, no retrying."""
    client, _ = writer
    assert client.get("/api/search?q=bespoke").json()["total"] == 0

    posted = client.post("/api/events", json={"events": [an_event()]})
    assert posted.status_code == 200
    assert posted.json()["searchable"] is True

    found = client.get("/api/search?q=bespoke").json()
    assert found["total"] == 1
    assert found["hits"][0]["document"]["title"] == "Bespoke Widget XR9"


def test_it_says_which_segment_the_event_landed_in(writer):
    client, _ = writer
    body = client.post("/api/events", json={"events": [an_event()]}).json()
    assert body["segment"].endswith(".seg")
    assert body["documents_in_segment"] == 1
    assert body["took_ms"] >= 0


def test_ingesting_does_not_disturb_what_was_already_there(writer):
    client, _ = writer
    before = client.get("/api/search?q=samsung").json()["total"]
    client.post("/api/events", json={"events": [an_event()]})
    assert client.get("/api/search?q=samsung").json()["total"] == before


def test_offsets_continue_rather_than_restarting(writer):
    """Manifest.publish evicts by offset range, so a writer that began again
    at zero would evict everything it had already published."""
    client, state = writer
    for n in range(3):
        client.post("/api/events", json={"events": [an_event(title=f"Widget {n}")]})
    assert client.get("/api/search?q=widget").json()["total"] == 3


def test_a_read_only_instance_refuses_to_ingest(store):
    state = ServiceState(store, manifest_prefix="manifests/", model_uri=None)
    state.load()
    reader = TestClient(create_app(state))
    assert reader.post("/api/events", json={"events": [an_event()]}).status_code == 503
    assert reader.get("/api/search?q=samsung").status_code == 200


def test_an_unknown_event_type_is_rejected(writer):
    client, _ = writer
    bad = an_event()
    bad["event_type"] = "teleport"
    assert client.post("/api/events", json={"events": [bad]}).status_code == 422


def test_an_oversized_batch_is_rejected(writer):
    from aether.service.app import MAX_INGEST_EVENTS

    client, _ = writer
    events = [an_event(title=f"w{n}") for n in range(MAX_INGEST_EVENTS + 1)]
    assert client.post("/api/events", json={"events": events}).status_code == 422


# --------------------------------------------------------------------------
# the feed
# --------------------------------------------------------------------------


def test_the_feed_round_trips_events(tmp_path, sample_csv):
    store = LocalStore(tmp_path / "f")
    chunks, total = write_feed(store, iter_events(sample_csv), chunk_events=10)
    assert chunks >= 2

    feed = Feed(store, chunk_events=10)
    seen = []
    while True:
        batch = feed.take(7)
        if not batch:
            break
        seen.extend(batch)
    assert len(seen) == total
    assert seen[0]["event_id"]


def test_the_feed_resumes_near_where_it_stopped(tmp_path, sample_csv):
    """Derived from the manifest rather than a stored cursor, so it rounds
    down to a chunk boundary and may replay a few."""
    store = LocalStore(tmp_path / "f")
    _, total = write_feed(store, iter_events(sample_csv), chunk_events=10)

    resumed = Feed(store, chunk_events=10, start_after=10)
    remaining = []
    while True:
        batch = resumed.take(10)
        if not batch:
            break
        remaining.extend(batch)
    assert len(remaining) == total - 10


def test_an_absent_feed_is_exhausted_not_an_error(tmp_path):
    feed = Feed(LocalStore(tmp_path / "empty"))
    assert feed.take(5) == []
    assert feed.exhausted


# --------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------


def test_streaming_indexes_and_reports_progress(store, sample_csv):
    write_feed(store, iter_events(sample_csv), chunk_events=5)
    state = ServiceState(
        store, manifest_prefix="manifests/", model_uri=None,
        refresh_seconds=0.0001, writable=True,
    )
    state.load()
    client = TestClient(create_app(state))

    before = client.get("/api/index").json()["documents"]
    with client.stream("GET", "/api/stream?seconds=1&rate=200") as response:
        assert response.status_code == 200
        events = [
            json.loads(line[len("data: "):])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert events, "expected at least a start and a done event"
    assert events[-1]["documents"] >= before
    assert client.get("/api/index").json()["documents"] > before


def test_streaming_without_a_feed_is_refused(store):
    state = ServiceState(
        store, manifest_prefix="manifests/", model_uri=None, writable=True
    )
    state.load()
    state.feed = None
    client = TestClient(create_app(state))
    assert client.get("/api/stream?seconds=1").status_code == 503


def test_the_stream_reports_the_whole_index_not_just_its_partition(store, sample_csv):
    """The live partition holds a few thousand of a much larger index. Sending
    its count as `documents` made the page's own header collapse the moment a
    burst started, which looked like data loss and was a naming mistake."""
    write_feed(store, iter_events(sample_csv), chunk_events=5)
    state = ServiceState(
        store, manifest_prefix="manifests/", model_uri=None,
        refresh_seconds=0.0001, writable=True,
    )
    state.load()
    client = TestClient(create_app(state))

    whole = client.get("/api/index").json()["documents"]
    assert state.ingestor.documents < whole, "fixture should already hold documents"

    with client.stream("GET", "/api/stream?seconds=1&rate=200") as response:
        payloads = [
            json.loads(line[len("data: "):])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert payloads
    for payload in payloads:
        if "documents" in payload:
            assert payload["documents"] >= whole
