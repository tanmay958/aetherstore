"""The query service.

Two things are served here, and they are the two halves of the project:
search over an index in object storage, and cart-abandonment prediction from
the model that storage also holds.

They share a process for a reason that is about deployment rather than
design. Both are stateless functions over an artifact in a bucket, both are
read-only, and both are wanted by the same dashboard. Splitting them would
double the cold starts and the deployment surface to separate two things
whose only shared resource is an HTTP port.

## Why this can scale to zero

Nothing here owns data. A fresh instance reads a manifest, and thereafter
reads only the byte ranges a query needs. There is no warm-up to protect, no
replica set to join, and no state to hand over on shutdown, so an instance
that has served nothing costs nothing to discard. Several instances behind a
load balancer share no coordination whatsoever, because segments are
immutable and nobody writes.

That is not free. A cold instance pays to open every segment it touches, and
that cost is a function of how many segments the index has, which is what
compaction exists to reduce. The `?explain=1` flag on a search reports it.

## Why /predict reuses the streaming predictor

`Predictor.handle` is the function the Kafka replicas run. This endpoint
drives the same object with the same events rather than reimplementing
scoring over HTTP, so the two cannot drift apart. A second implementation
here would be the classic train/serve skew bug wearing a different hat, and
there is a test asserting the two paths agree.
"""

from __future__ import annotations

import os
import secrets
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from aether.events import EVENT_TYPES
from aether.index.analyzer import tokenize
from aether.service.state import ServiceState, state_from_env

# When set, every /api/* request must carry this in `x-aether-key`. It is
# held by the Cloudflare Pages Function in front of the service, never by the
# page, because anything the page knows is in the visitor's network tab.
#
# Unset means open, which is the honest default for a demo whose data is a
# public dataset: the thing worth protecting here is the bill, and that is
# what the query caps below do. This closes the door on casual scripted abuse
# once the proxy is in place; it is a shared secret, not an identity, and it
# does not pretend to be one.
API_KEY_ENV = "AETHER_API_KEY"
API_KEY_HEADER = "x-aether-key"

MAX_EVENTS_PER_REQUEST = 500
MAX_TOP_K = 100

# A query costs roughly one dictionary read plus one postings read per term
# per segment, so its cost is linear in the number of terms and there is no
# natural ceiling on how many a caller can send. Measured against the
# 1,000,000-document index: one term cost 100 object storage requests, and 267
# terms cost 27,115. That is a 271x amplification of one cheap HTTP request
# into someone else's storage bill, and roughly 370 such requests would exhaust
# a month of R2's free tier.
#
# Nobody searches for sixteen words. This is not a limit real use will reach,
# and it is what keeps the cost of a request bounded by the request rather than
# by its contents. Rate limiting belongs at the edge; this belongs here,
# because it is a property of the engine and not of the deployment.
MAX_QUERY_TERMS = 16
MAX_QUERY_CHARS = 256


# --------------------------------------------------------------------------
# request and response shapes
# --------------------------------------------------------------------------


class EventIn(BaseModel):
    """One event of a session, as the dashboard would send it.

    Deliberately loose about identity and strict about type: a caller
    exploring the model should not have to invent event ids, but an unknown
    event type is a bug worth surfacing rather than silently ignoring.
    """

    ts: int
    event_type: str
    session_id: str = "http-session"
    user_id: str = ""
    device: str | None = None
    product_id: str | None = None
    title: str | None = None
    category: str | None = None
    brand: str | None = None
    price: float | None = None
    query: str | None = None

    def as_event(self, index: int) -> dict:
        if self.event_type not in EVENT_TYPES:
            raise HTTPException(
                422, f"unknown event_type {self.event_type!r}; expected one of "
                     f"{sorted(EVENT_TYPES)}"
            )
        event = self.model_dump()
        event["event_id"] = f"http-{index}"
        return event


class PredictIn(BaseModel):
    events: list[EventIn] = Field(min_length=1)
    threshold: float = 0.7


class Score(BaseModel):
    """The prediction after one event, or why there wasn't one."""

    index: int
    ts: int
    event_type: str
    events: int
    cart_size: int
    cart_value: float
    probability: float | None
    scored: bool
    reason: str | None = None


def create_app(state: ServiceState | None = None) -> FastAPI:
    """Build the app. Takes state so tests can supply a local index."""
    app = FastAPI(
        title="AetherStore",
        description="Distributed search and streaming prediction, from scratch.",
        version="0.1.0",
    )

    expected_key = os.environ.get(API_KEY_ENV) or None

    @app.middleware("http")
    async def require_key(request: Request, call_next):
        """Gate /api/* on a shared secret, when one is configured.

        `/health` is deliberately outside the gate: an uptime check should not
        need a credential, and it discloses nothing but whether the index and
        model loaded.
        """
        if expected_key and request.url.path.startswith("/api/"):
            offered = request.headers.get(API_KEY_HEADER, "")
            # Constant time, so a mismatch cannot be found one byte at a time.
            if not secrets.compare_digest(offered, expected_key):
                return JSONResponse({"detail": "not authorised"}, status_code=401)
        return await call_next(request)

    resolved: dict[str, ServiceState] = {}

    def current() -> ServiceState:
        # Loaded on first use rather than at import, so constructing the app
        # never touches the network and a test can inject its own.
        if "state" not in resolved:
            resolved["state"] = state if state is not None else state_from_env()
        return resolved["state"]

    if state is not None:
        resolved["state"] = state

    # -- liveness ----------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Deliberately reports degraded rather than failing.

        A container that exits because a model is missing tells an operator
        nothing and stops search working too. This says what is wrong.
        """
        service = current()
        return {
            "status": service.status,
            "index": service.index_summary(),
            "model": {"loaded": service.model is not None, "error": service.model_error},
        }

    @app.get("/api/index")
    def index_info() -> dict[str, Any]:
        return current().index_summary()

    @app.get("/api/model")
    def model_info() -> dict[str, Any]:
        return current().model_summary()

    # -- search ------------------------------------------------------------

    @app.get("/api/search")
    def search(
        q: str = Query(
            min_length=1,
            max_length=MAX_QUERY_CHARS,
            description="the query",
        ),
        k: int = Query(10, ge=1, le=MAX_TOP_K),
        mode: Literal["and", "or"] = "and",
        global_stats: bool = Query(
            False,
            description="score every segment against one corpus, at the cost "
                        "of a second round trip",
        ),
        explain: bool = Query(
            False, description="report what this query cost in requests and bytes"
        ),
    ) -> dict[str, Any]:
        service = current()
        if not service.ready:
            raise HTTPException(503, f"index unavailable: {service.index_error}")

        # Counted after analysis, because that is what the engine will
        # actually go and read. Counting the raw string would let punctuation
        # and stopwords through, and would reject queries that cost nothing.
        terms = tokenize(q)
        if len(terms) > MAX_QUERY_TERMS:
            raise HTTPException(
                422,
                f"{len(terms)} search terms exceeds the {MAX_QUERY_TERMS} "
                "allowed: each term costs a read in every segment, so an "
                "unbounded query is an unbounded bill",
            )

        def run() -> tuple[Any, list[dict]]:
            result = service.coordinator.search(
                q, top_k=k, mode=mode, global_stats=global_stats
            )
            return result, service.coordinator.documents(result.hits)

        if explain:
            with service.measured() as box:
                result, documents = run()
            cost = box[0]
        else:
            result, documents = run()
            cost = None

        body: dict[str, Any] = {
            "query": q,
            "total": result.total,
            "took_ms": round(result.stats.elapsed_ms, 1),
            "hits": [
                {
                    "score": round(hit.score, 6),
                    "segment": hit.segment,
                    "doc_id": hit.doc_id,
                    "document": document,
                }
                for hit, document in zip(result.hits, documents)
            ],
        }
        if cost is not None:
            body["cost"] = {
                "requests": cost.requests,
                "bytes": cost.bytes_read,
                "segments_searched": result.stats.segments_searched,
                "segments_pruned": result.stats.segments_pruned,
                "segments_total": result.stats.segments_total,
                "waves": result.stats.waves,
                "note": "a warm instance has already paid to open these "
                        "segments; a cold one pays two requests each",
            }
        return body

    # -- prediction --------------------------------------------------------

    @app.post("/api/predict")
    def predict(body: PredictIn) -> dict[str, Any]:
        """Score a session event by event.

        Returns the whole trajectory rather than one number, because the
        interesting thing about this model is how the probability moves as a
        shopper acts, and because it is what the streaming replicas emit.
        """
        service = current()
        if service.model is None:
            raise HTTPException(503, f"model unavailable: {service.model_error}")
        if len(body.events) > MAX_EVENTS_PER_REQUEST:
            raise HTTPException(
                413, f"{len(body.events)} events exceeds the "
                     f"{MAX_EVENTS_PER_REQUEST} allowed in one request"
            )

        from aether.stream.config import KafkaConfig
        from aether.stream.predictor import Predictor

        # The same object the Kafka replicas run, driven by hand. No Kafka is
        # contacted: `handle` is a pure function of the event and the state.
        predictor = Predictor(
            service.model,
            KafkaConfig(),
            output_topic=None,
            risk_threshold=body.threshold,
        )

        scores: list[Score] = []
        for index, incoming in enumerate(body.events):
            event = incoming.as_event(index)
            prediction = predictor.handle(event)
            state = predictor.sessions[event["session_id"]]
            scores.append(
                Score(
                    index=index,
                    ts=event["ts"],
                    event_type=event["event_type"],
                    events=state.events,
                    cart_size=state.cart_size,
                    cart_value=round(state.cart_value, 2),
                    probability=prediction.probability if prediction else None,
                    scored=prediction is not None,
                    reason=None if prediction else _why_not(state),
                )
            )

        scored = [s for s in scores if s.scored]
        return {
            "threshold": body.threshold,
            "scores": [s.model_dump() for s in scores],
            "final": scored[-1].probability if scored else None,
            "at_risk": bool(scored and scored[-1].probability >= body.threshold),
        }

    return app


def _why_not(state) -> str:
    """Why an event produced no prediction, in the caller's terms."""
    from aether.stream.predictor import MIN_EVENTS_TO_SCORE

    if not state.has_open_cart:
        return (
            "no open cart: a purchase settles the cart, and a session with "
            "nothing in one cannot abandon it"
        )
    if state.events < MIN_EVENTS_TO_SCORE:
        return f"fewer than {MIN_EVENTS_TO_SCORE} events so far"
    return "not scored"


app = create_app()
