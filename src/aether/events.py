"""The canonical event schema.

Everything downstream of ingestion reads events in exactly this shape: the
analyzer, the indexer, the segment writer, and later the ML feature builder.
Keeping one definition here means a schema change touches one file, and that
adding a second data source costs a normalizer rather than a refactor.

Fields are nullable because the source data genuinely lacks some of them.
REES46 carries no search events and no device information, so `query` and
`device` are always None for it. Nothing here invents a value to fill a gap.
The single exception is `title`, and `aether.data.titles` documents exactly
what is real in it and what is derived.
"""

from __future__ import annotations

# Canonical key order. The segment writer depends on a stable schema, so this
# tuple is the single definition of it and every source builds against it.
EVENT_FIELDS = (
    "event_id",
    "ts",
    "session_id",
    "user_id",
    "event_type",
    "device",
    "product_id",
    "title",
    "category",
    "brand",
    "price",
    "query",
)

# Fields whose text belongs in the inverted index. Identifiers and numbers are
# for filtering and display, which the doc store handles instead.
TEXT_FIELDS = ("title", "category", "brand", "query", "event_type")

# The shared event vocabulary. REES46's "cart" is normalized to "add_to_cart"
# so a downstream consumer never has to know which source produced an event.
EVENT_TYPES = frozenset(
    {"view", "click", "search", "add_to_cart", "remove_from_cart", "purchase"}
)

# The point at which a session becomes capable of being abandoned. Both the
# abandonment label and the funnel statistics key off these two.
CART_EVENT = "add_to_cart"
PURCHASE_EVENT = "purchase"


def make_event(
    *,
    event_id: str,
    ts: int,
    session_id: str,
    user_id: str,
    event_type: str,
    device: str | None = None,
    product_id: str | None = None,
    title: str | None = None,
    category: str | None = None,
    brand: str | None = None,
    price: float | None = None,
    query: str | None = None,
) -> dict:
    """Build one canonical event.

    Keyword-only, so a caller cannot silently transpose two string fields, and
    the result always has keys in `EVENT_FIELDS` order.
    """
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown event_type {event_type!r}")
    return {
        "event_id": event_id,
        "ts": ts,
        "session_id": session_id,
        "user_id": user_id,
        "event_type": event_type,
        "device": device,
        "product_id": product_id,
        "title": title,
        "category": category,
        "brand": brand,
        "price": price,
        "query": query,
    }


def validate_event(event: dict) -> None:
    """Raise if an event does not match the canonical schema.

    Cheap enough to call per event in a test, deliberately not called in the
    loader's hot path where it would cost real time across tens of millions
    of rows.
    """
    keys = tuple(event)
    if keys != EVENT_FIELDS:
        raise ValueError(f"event keys {keys} do not match schema {EVENT_FIELDS}")
    if event["event_type"] not in EVENT_TYPES:
        raise ValueError(f"unknown event_type {event['event_type']!r}")
    if not isinstance(event["ts"], int):
        raise ValueError(f"ts must be an int epoch, got {type(event['ts']).__name__}")
