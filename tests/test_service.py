"""Tests for the HTTP query service.

Most of these check the shape of an endpoint. Two check something worth more
than that.

`test_http_and_streaming_agree_exactly` is the important one. The endpoint
and the Kafka replicas must produce identical probabilities, because they are
supposed to be the same code, and the moment someone reimplements scoring for
HTTP this fails.

`test_explain_reports_what_the_query_actually_cost` checks the cost numbers
against the store's own counters rather than trusting the endpoint to report
them honestly.
"""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from aether.data.rees46 import iter_events  # noqa: E402
from aether.index.ingest import ingest  # noqa: E402
from aether.service.app import create_app  # noqa: E402
from aether.service.state import ServiceState  # noqa: E402
from aether.storage import LocalStore  # noqa: E402

BASE_TS = 1_570_000_000


@pytest.fixture
def indexed(tmp_path, sample_csv):
    """A real index, split so the coordinator has several segments to merge."""
    store = LocalStore(tmp_path / "idx")
    ingest(iter_events(sample_csv), store, docs_per_segment=8)
    return store


@pytest.fixture
def model(tmp_path, sample_csv):
    """A model trained on the fixture. Tiny and bad, which does not matter:
    these tests are about plumbing, not accuracy."""
    pytest.importorskip("sklearn")
    import time

    from aether.ml.dataset import build_dataset
    from aether.ml.features import FEATURE_NAMES
    from aether.ml.model import ModelArtifact
    from aether.ml.train import fit_model

    data = build_dataset(iter_events(sample_csv))
    artifact = ModelArtifact(
        estimator=fit_model(data),
        feature_names=FEATURE_NAMES,
        metrics={"model": {"note": "fixture"}},
        trained_at=time.time(),
        trained_on_events=len(data),
    )
    path = tmp_path / "model.pkl"
    artifact.save(path)
    return path


@pytest.fixture
def client(indexed, model):
    state = ServiceState(indexed, model_uri=str(model))
    state.load()
    return TestClient(create_app(state))


@pytest.fixture
def client_without_model(indexed):
    state = ServiceState(indexed, model_uri=None)
    state.load()
    return TestClient(create_app(state))


def session_events(with_cart: bool = True) -> list[dict]:
    events = [
        {"ts": BASE_TS, "event_type": "view", "product_id": "a", "price": 30.0,
         "brand": "samsung", "category": "electronics.smartphone"},
        {"ts": BASE_TS + 12, "event_type": "view", "product_id": "b", "price": 90.0,
         "brand": "apple", "category": "electronics.smartphone"},
    ]
    if with_cart:
        events.append(
            {"ts": BASE_TS + 30, "event_type": "add_to_cart", "product_id": "b",
             "price": 90.0, "brand": "apple", "category": "electronics.smartphone"}
        )
        events.append(
            {"ts": BASE_TS + 55, "event_type": "view", "product_id": "c", "price": 15.0,
             "brand": "xiaomi", "category": "electronics.audio"}
        )
    return events


# --------------------------------------------------------------------------
# liveness
# --------------------------------------------------------------------------


def test_health_reports_the_index_and_the_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["index"]["documents"] > 0
    assert body["index"]["segments"] > 1
    assert body["model"]["loaded"] is True


def test_a_missing_model_degrades_rather_than_kills_the_service(client_without_model):
    """A container that exits because a model is missing cannot serve search
    either, and tells an operator nothing."""
    body = client_without_model.get("/health").json()
    assert body["model"]["loaded"] is False
    assert client_without_model.get("/api/search?q=samsung").status_code == 200
    assert client_without_model.post(
        "/api/predict", json={"events": session_events()}
    ).status_code == 503


def test_an_empty_index_is_healthy_but_says_so(tmp_path):
    """An absent manifest is a valid empty index, not an error. But a service
    pointed at the wrong bucket looks identical, so it must not report `ok`."""
    state = ServiceState(LocalStore(tmp_path / "empty"), model_uri=None)
    state.load()
    client = TestClient(create_app(state))
    assert client.get("/health").json()["status"] == "empty"
    assert client.get("/api/search?q=samsung").json()["total"] == 0


def test_an_unreachable_index_degrades_rather_than_raises(tmp_path):
    """A bad endpoint or bad credentials must be reported by /health, not
    thrown as a 500 at whoever queries first."""

    class Unreachable(LocalStore):
        def get_range(self, key, start, length):
            raise ConnectionError("no route to host")

        def size(self, key):
            raise ConnectionError("no route to host")

    state = ServiceState(Unreachable(tmp_path / "gone"), model_uri=None)
    state.load()
    client = TestClient(create_app(state))
    assert client.get("/health").json()["status"] == "degraded"
    assert client.get("/api/search?q=samsung").status_code == 503


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------


def test_search_returns_ranked_documents(client):
    body = client.get("/api/search?q=samsung smartphone&k=5").json()
    assert body["total"] > 0
    assert body["hits"]
    scores = [hit["score"] for hit in body["hits"]]
    assert scores == sorted(scores, reverse=True)
    assert all(hit["document"]["event_id"] for hit in body["hits"])


def test_search_matches_the_coordinator_it_wraps(client, indexed):
    """The endpoint must not quietly re-rank."""
    from aether.index.coordinator import Coordinator

    expected = Coordinator(indexed).search("samsung smartphone", top_k=5)
    body = client.get("/api/search?q=samsung smartphone&k=5").json()
    assert body["total"] == expected.total
    assert [hit["doc_id"] for hit in body["hits"]] == [h.doc_id for h in expected.hits]


def test_or_mode_matches_at_least_as_much_as_and(client):
    conjunction = client.get("/api/search?q=samsung smartphone&mode=and").json()
    disjunction = client.get("/api/search?q=samsung smartphone&mode=or").json()
    assert disjunction["total"] >= conjunction["total"]


def test_global_stats_costs_a_second_wave(client):
    body = client.get("/api/search?q=samsung smartphone&global_stats=true&explain=true").json()
    assert body["cost"]["waves"] == 2


def test_cost_is_absent_unless_asked_for(client):
    """It takes a lock, so it must never be on the default path."""
    assert "cost" not in client.get("/api/search?q=samsung").json()


def test_explain_reports_what_the_query_actually_cost(client, indexed):
    """Checked against the store's own counters, not taken on trust."""
    from aether.storage import CountingStore

    counting = CountingStore(indexed)
    state = ServiceState(counting, model_uri=None)
    state.load()
    local = TestClient(create_app(state))
    # Warm, so the measured query pays for the search rather than for opening.
    local.get("/api/search?q=samsung")

    before = counting.stats.requests
    body = local.get("/api/search?q=samsung smartphone&explain=true").json()
    actually_spent = counting.stats.requests - before

    assert body["cost"]["requests"] == actually_spent
    assert body["cost"]["bytes"] > 0
    assert body["cost"]["segments_total"] >= body["cost"]["segments_searched"]


def test_an_empty_query_is_rejected(client):
    assert client.get("/api/search?q=").status_code == 422


def test_top_k_is_bounded(client):
    """An unbounded k is a way to ask one request to read a whole index."""
    assert client.get("/api/search?q=samsung&k=100000").status_code == 422


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------


def test_predict_returns_a_trajectory_not_a_number(client):
    body = client.post("/api/predict", json={"events": session_events()}).json()
    assert len(body["scores"]) == 4
    assert [s["scored"] for s in body["scores"]] == [False, False, True, True]
    assert 0.0 <= body["final"] <= 1.0


def test_events_before_a_cart_are_not_scored(client):
    body = client.post("/api/predict", json={"events": session_events(with_cart=False)}).json()
    assert body["final"] is None
    assert all(not s["scored"] for s in body["scores"])
    assert "no open cart" in body["scores"][0]["reason"]


def test_a_purchase_settles_the_cart_and_scoring_stops(client):
    events = session_events()
    events.append({"ts": BASE_TS + 90, "event_type": "purchase", "product_id": "b",
                   "price": 90.0})
    body = client.post("/api/predict", json={"events": events}).json()
    last = body["scores"][-1]
    assert last["scored"] is False
    assert last["cart_size"] == 0
    # The session itself carries on, with its history intact.
    assert last["events"] == 5


def test_at_risk_follows_the_threshold(client):
    events = session_events()
    never = client.post("/api/predict", json={"events": events, "threshold": 1.0}).json()
    always = client.post("/api/predict", json={"events": events, "threshold": 0.0}).json()
    assert never["at_risk"] is False
    assert always["at_risk"] is True


def test_an_unknown_event_type_is_rejected(client):
    events = [{"ts": BASE_TS, "event_type": "teleport"}]
    assert client.post("/api/predict", json={"events": events}).status_code == 422


def test_an_oversized_session_is_rejected(client):
    from aether.service.app import MAX_EVENTS_PER_REQUEST

    events = [
        {"ts": BASE_TS + n, "event_type": "view", "product_id": "a"}
        for n in range(MAX_EVENTS_PER_REQUEST + 1)
    ]
    assert client.post("/api/predict", json={"events": events}).status_code == 413


def test_an_empty_session_is_rejected(client):
    assert client.post("/api/predict", json={"events": []}).status_code == 422


# --------------------------------------------------------------------------
# the one that keeps HTTP and Kafka honest
# --------------------------------------------------------------------------


def test_http_and_streaming_agree_exactly(client, model):
    """The endpoint drives `Predictor.handle`, the same function the Kafka
    replicas run. If anyone reimplements scoring for HTTP, this fails."""
    from aether.ml.model import ModelArtifact
    from aether.stream.config import KafkaConfig
    from aether.stream.predictor import Predictor

    events = session_events()
    body = client.post("/api/predict", json={"events": events}).json()

    predictor = Predictor(ModelArtifact.load(model), KafkaConfig(), output_topic=None)
    expected = []
    for index, event in enumerate(events):
        full = {"event_id": f"http-{index}", "session_id": "http-session",
                "user_id": "", "device": None, "title": None, "query": None,
                "category": None, "brand": None, "price": None, "product_id": None}
        full.update(event)
        prediction = predictor.handle(full)
        expected.append(prediction.probability if prediction else None)

    assert [s["probability"] for s in body["scores"]] == expected


def test_each_request_starts_from_a_clean_session(client):
    """State must not leak between callers: two identical requests must give
    identical answers, which they cannot if the first left a session behind."""
    events = session_events()
    first = client.post("/api/predict", json={"events": events}).json()
    second = client.post("/api/predict", json={"events": events}).json()
    assert first == second
