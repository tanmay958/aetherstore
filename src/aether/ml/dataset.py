"""Turning a stream of events into labelled training examples.

This file is where a cart-abandonment model is most easily got wrong, and
every choice below is guarding against one specific way of getting it wrong.

**What is being predicted.** Given what a session has done so far, will it end
without buying? Only sessions that put something in a cart are considered: a
session that never carted cannot abandon a cart, and including them would
train the model to predict "did they shop at all", which is a different and
much easier question that looks great in the metrics and is useless.

**The unit is a cart, not a session.** Real REES46 sessions routinely contain
several purchases: a shopper buys, keeps browsing, carts again, and buys
again. One observed session has four purchases across thirty-seven events.
Labelling per session would call that whole session "converted" and throw away
every later cart, including any the shopper genuinely abandoned.

So a session is split into cycles. A cycle opens at the first `add_to_cart`
after the last purchase, and closes either at the next purchase, which makes
it converted, or at the end of the session, which makes it abandoned.

**When examples are taken.** One per event while a cart is open. Before the
first add there is nothing to abandon. After a purchase the cart is settled
and the answer for that cycle is already known, so the next example waits for
the next add.

**The label comes from the whole session; the features never do.** The label
is known only in hindsight, which is fine because labelling happens offline.
The features at each example are the state as it stood at that moment, which
is exactly what serving will have. A snapshot is taken rather than a
reference, because the live state object keeps mutating.

**Splitting by time, not at random.** Sessions overlap and shoppers behave
differently on a Tuesday morning than during a sale. A random split puts
neighbouring sessions on both sides, so the model gets to see the period it is
being tested on and scores far better than it deserves. Splitting on session
start time reproduces the only situation that matters: predicting a future
nobody has seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np

from aether.events import CART_EVENT, PURCHASE_EVENT
from aether.ml.features import FEATURE_COUNT, compute_features
from aether.ml.session import SessionState


@dataclass
class Example:
    """One training row: what was known, and what happened in the end."""

    features: np.ndarray
    abandoned: bool
    session_id: str
    session_start: int
    event_index: int


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    session_ids: np.ndarray
    session_starts: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    @property
    def abandonment_rate(self) -> float:
        return float(self.y.mean()) if len(self.y) else 0.0

    @property
    def sessions(self) -> int:
        return len(set(self.session_ids.tolist()))


@dataclass
class BuildStats:
    events: int = 0
    sessions: int = 0
    sessions_with_cart: int = 0
    sessions_converted: int = 0
    examples: int = 0
    dropped_no_cart: int = 0
    per_session_examples: list[int] = field(default_factory=list)

    def __str__(self) -> str:
        rate = (
            100 * (self.sessions_with_cart - self.sessions_converted) / self.sessions_with_cart
            if self.sessions_with_cart
            else 0.0
        )
        return (
            f"{self.events:,} events, {self.sessions:,} sessions, "
            f"{self.sessions_with_cart:,} with a cart of which "
            f"{self.sessions_converted:,} converted ({rate:.1f}% abandoned), "
            f"{self.examples:,} examples"
        )


def group_by_session(events: Iterable[dict]) -> Iterator[list[dict]]:
    """Collect events into whole sessions.

    Buffers, because a session's label depends on whether it ever purchased
    and that is only known at its end. Sessions interleave in the stream, so
    this holds every open session in memory: fine for training over a slice,
    and precisely the thing the streaming predictor cannot do, which is why it
    predicts continuously instead of labelling.
    """
    sessions: dict[str, list[dict]] = {}
    for event in events:
        sessions.setdefault(event["session_id"], []).append(event)
    for grouped in sessions.values():
        grouped.sort(key=lambda event: event["ts"])
        yield grouped


def examples_from_session(events: list[dict]) -> list[Example]:
    """Every labelled example a session yields, one cart cycle at a time.

    Empty for a session that never put anything in a cart.
    """
    if not any(event["event_type"] == CART_EVENT for event in events):
        return []

    state = SessionState(events[0]["session_id"])
    start = events[0]["ts"]

    out: list[Example] = []
    # Examples for the cart currently open. Their label is unknown until the
    # cycle closes, so they are held here and labelled on the way out.
    pending: list[Example] = []

    def close_cycle(converted: bool) -> None:
        nonlocal pending
        for example in pending:
            out.append(
                Example(
                    example.features,
                    not converted,
                    example.session_id,
                    example.session_start,
                    example.event_index,
                )
            )
        pending = []

    for index, event in enumerate(events):
        if event["event_type"] == PURCHASE_EVENT:
            # The open cart converted. Fold the event in afterwards, so the
            # state carries on for the rest of the session with a clean cart.
            close_cycle(converted=True)
            state.update(event)
            continue

        state.update(event)
        if not state.has_open_cart:
            continue
        pending.append(
            Example(compute_features(state.copy()), True, state.session_id, start, index)
        )

    # Whatever was still in the cart when the session ended was abandoned.
    close_cycle(converted=False)
    return out


def build_dataset(
    events: Iterable[dict], stats: BuildStats | None = None
) -> Dataset:
    """Labelled examples from a stream of events."""
    stats = stats if stats is not None else BuildStats()
    rows: list[Example] = []

    materialised = list(events)
    stats.events = len(materialised)

    for session_events in group_by_session(materialised):
        stats.sessions += 1
        session_examples = examples_from_session(session_events)
        if not session_examples:
            stats.dropped_no_cart += 1
            continue
        stats.sessions_with_cart += 1
        # A session counts as converted if any of its cart cycles did.
        if any(not example.abandoned for example in session_examples):
            stats.sessions_converted += 1
        stats.per_session_examples.append(len(session_examples))
        rows.extend(session_examples)

    stats.examples = len(rows)
    if not rows:
        empty = np.zeros((0, FEATURE_COUNT))
        return Dataset(empty, np.zeros(0, dtype=bool), np.array([]), np.zeros(0, dtype=np.int64))

    return Dataset(
        np.vstack([row.features for row in rows]),
        np.array([row.abandoned for row in rows], dtype=bool),
        np.array([row.session_id for row in rows]),
        np.array([row.session_start for row in rows], dtype=np.int64),
    )


def time_split(data: Dataset, train_fraction: float = 0.7) -> tuple[Dataset, Dataset]:
    """Split on session start time, keeping whole sessions on one side.

    Two mistakes are avoided here at once. Splitting at random would let the
    model see the period it is tested on. Splitting rows rather than sessions
    would put early events of a session in training and later ones in test,
    where they share a label and most of their feature values, which is
    leakage wearing a time-shaped disguise.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if len(data) == 0:
        return data, data

    starts = sorted({int(start) for start in data.session_starts.tolist()})
    cutoff = starts[max(0, int(len(starts) * train_fraction) - 1)]

    is_train = data.session_starts <= cutoff
    return (
        Dataset(
            data.X[is_train],
            data.y[is_train],
            data.session_ids[is_train],
            data.session_starts[is_train],
        ),
        Dataset(
            data.X[~is_train],
            data.y[~is_train],
            data.session_ids[~is_train],
            data.session_starts[~is_train],
        ),
    )
