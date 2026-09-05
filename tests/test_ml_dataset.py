"""Tests for how training examples are built.

Almost every test here is guarding against one specific way of leaking the
answer into the question. A cart-abandonment model that leaks scores
beautifully offline and is worthless live, and the failure is silent: nothing
crashes, the metrics simply lie.
"""

import numpy as np
import pytest

from aether.ml.dataset import (
    build_dataset,
    examples_from_session,
    group_by_session,
    time_split,
)

BASE_TS = 1_570_000_000


def event(kind: str, ts: int, session: str = "s1", price: float = 10.0) -> dict:
    return {
        "event_id": f"{session}-{ts}",
        "ts": ts,
        "session_id": session,
        "user_id": "u1",
        "event_type": kind,
        "device": "mobile",
        "product_id": "p1",
        "title": "thing",
        "category": "electronics.phone",
        "brand": "samsung",
        "price": price,
        "query": None,
    }


def session(kinds: list[str], name: str = "s1", start: int = BASE_TS) -> list[dict]:
    return [event(kind, start + i * 10, name) for i, kind in enumerate(kinds)]


# --------------------------------------------------------------------------
# which sessions count
# --------------------------------------------------------------------------


def test_a_session_that_never_carted_yields_nothing():
    """It cannot abandon a cart it never made. Including such sessions would
    train the model to predict "did they shop at all", a different and much
    easier question that flatters the metrics and answers nothing."""
    assert examples_from_session(session(["view", "view", "view"])) == []


def test_a_carted_session_yields_examples():
    assert examples_from_session(session(["view", "add_to_cart", "view"]))


def test_examples_begin_at_the_first_cart_not_before():
    """Before the cart exists there is nothing to abandon."""
    events = session(["view", "view", "add_to_cart", "view"])
    examples = examples_from_session(events)
    assert [e.event_index for e in examples] == [2, 3]


def test_examples_stop_before_a_purchase():
    """Past the purchase the state contains the answer."""
    events = session(["view", "add_to_cart", "view", "purchase", "view"])
    examples = examples_from_session(events)
    assert [e.event_index for e in examples] == [1, 2]


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------


def test_a_converted_session_is_labelled_not_abandoned():
    examples = examples_from_session(session(["add_to_cart", "view", "purchase"]))
    assert examples
    assert all(not e.abandoned for e in examples)


def test_an_abandoned_session_is_labelled_abandoned():
    examples = examples_from_session(session(["add_to_cart", "view", "view"]))
    assert examples
    assert all(e.abandoned for e in examples)


def test_the_label_is_the_same_for_every_example_in_a_session():
    """It describes the session's outcome, not the moment."""
    examples = examples_from_session(session(["add_to_cart"] + ["view"] * 8))
    assert len({e.abandoned for e in examples}) == 1


def test_a_removal_does_not_make_a_session_converted():
    examples = examples_from_session(
        session(["add_to_cart", "remove_from_cart", "view"])
    )
    assert all(e.abandoned for e in examples)


# --------------------------------------------------------------------------
# grouping
# --------------------------------------------------------------------------


def test_interleaved_sessions_are_separated():
    events = [
        event("view", BASE_TS, "a"),
        event("view", BASE_TS + 1, "b"),
        event("add_to_cart", BASE_TS + 2, "a"),
        event("add_to_cart", BASE_TS + 3, "b"),
    ]
    grouped = {g[0]["session_id"]: g for g in group_by_session(events)}
    assert set(grouped) == {"a", "b"}
    assert all(len(g) == 2 for g in grouped.values())


def test_events_are_ordered_within_a_session():
    """The stream is not globally sorted by time, and folding events out of
    order would compute nonsense gaps."""
    events = [
        event("add_to_cart", BASE_TS + 50, "a"),
        event("view", BASE_TS, "a"),
        event("view", BASE_TS + 20, "a"),
    ]
    grouped = next(iter(group_by_session(events)))
    assert [e["ts"] for e in grouped] == sorted(e["ts"] for e in grouped)


# --------------------------------------------------------------------------
# the dataset
# --------------------------------------------------------------------------


def test_counts_are_reported():
    events = (
        session(["view", "add_to_cart", "view"], "a")
        + session(["view", "view"], "b")
        + session(["add_to_cart", "purchase"], "c")
    )
    from aether.ml.dataset import BuildStats

    stats = BuildStats()
    data = build_dataset(events, stats)

    assert stats.sessions == 3
    assert stats.sessions_with_cart == 2
    assert stats.sessions_converted == 1
    assert stats.dropped_no_cart == 1
    assert len(data) == stats.examples


def test_an_empty_stream_produces_an_empty_dataset():
    data = build_dataset([])
    assert len(data) == 0
    assert data.abandonment_rate == 0.0


def test_features_are_finite():
    """A single NaN poisons a whole training batch."""
    events = session(["view", "add_to_cart", "view", "remove_from_cart"], "a")
    assert np.all(np.isfinite(build_dataset(events).X))


# --------------------------------------------------------------------------
# splitting, where the subtlest leakage lives
# --------------------------------------------------------------------------


def build_many(count: int = 20):
    events = []
    for i in range(count):
        kinds = ["view", "add_to_cart", "view"] + (["purchase"] if i % 2 else [])
        events += session(kinds, f"s{i:03d}", BASE_TS + i * 10_000)
    return build_dataset(events)


def test_no_session_appears_on_both_sides():
    """Splitting rows rather than sessions would put a session's early events
    in training and its later ones in test, where they share a label and most
    of their feature values. That is leakage wearing a time-shaped disguise."""
    train, test = time_split(build_many(), 0.7)
    assert not set(train.session_ids.tolist()) & set(test.session_ids.tolist())


def test_the_test_set_is_strictly_later():
    """A random split lets the model see the period it is judged on."""
    train, test = time_split(build_many(), 0.7)
    assert train.session_starts.max() < test.session_starts.min()


def test_the_split_is_roughly_where_it_was_asked_to_be():
    train, test = time_split(build_many(100), 0.7)
    fraction = train.sessions / (train.sessions + test.sessions)
    assert 0.6 < fraction < 0.8


def test_both_sides_keep_both_labels():
    """A split that put every conversion on one side would produce a model
    that cannot be evaluated."""
    train, test = time_split(build_many(40), 0.7)
    assert 0 < train.y.mean() < 1
    assert 0 < test.y.mean() < 1


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5, 1.5])
def test_rejects_a_nonsense_fraction(fraction):
    with pytest.raises(ValueError, match="train_fraction"):
        time_split(build_many(4), fraction)


def test_splitting_nothing_is_not_an_error():
    empty = build_dataset([])
    train, test = time_split(empty)
    assert len(train) == 0 and len(test) == 0
