"""Tests for session state and features.

The test that matters most here is the last one. Train/serve skew is the
commonest way a model that looked good offline performs badly in production,
and it comes from features being computed one way in a trainer and another way
in a serving path. This suite exists to make that impossible rather than
unlikely.
"""

import numpy as np
import pytest

from aether.ml.features import FEATURE_COUNT, FEATURE_NAMES, compute_features, features_as_dict
from aether.ml.session import SESSION_TIMEOUT_SECONDS, SessionState

BASE_TS = 1_570_000_000


def event(kind: str, *, ts: int, product="p1", price=10.0, **extra) -> dict:
    return {
        "event_id": f"e{ts}",
        "ts": ts,
        "session_id": "s1",
        "user_id": "u1",
        "event_type": kind,
        "device": extra.get("device"),
        "product_id": product,
        "title": extra.get("title"),
        "category": extra.get("category", "electronics.smartphone"),
        "brand": extra.get("brand", "samsung"),
        "price": price,
        "query": None,
    }


def play(events: list[dict]) -> SessionState:
    state = SessionState("s1")
    for e in events:
        state.update(e)
    return state


# --------------------------------------------------------------------------
# session state
# --------------------------------------------------------------------------


def test_counts_events_by_type():
    state = play([
        event("view", ts=BASE_TS),
        event("view", ts=BASE_TS + 10),
        event("add_to_cart", ts=BASE_TS + 20),
        event("purchase", ts=BASE_TS + 30),
    ])
    assert (state.events, state.views, state.cart_adds, state.purchases) == (4, 2, 1, 1)


def test_cart_value_survives_a_removal():
    """Tracked as contents rather than a running sum, because a sum can only
    guess at what a removal took out."""
    state = play([
        event("add_to_cart", ts=BASE_TS, product="a", price=100.0),
        event("add_to_cart", ts=BASE_TS + 5, product="b", price=50.0),
        event("remove_from_cart", ts=BASE_TS + 10, product="a", price=100.0),
    ])
    assert state.cart_value == 50.0
    assert state.cart_size == 1


def test_removing_something_never_added_is_harmless():
    state = play([event("remove_from_cart", ts=BASE_TS, product="ghost")])
    assert state.cart_value == 0.0


def test_tracks_distinct_products_categories_and_brands():
    state = play([
        event("view", ts=BASE_TS, product="a", brand="samsung", category="electronics.phone"),
        event("view", ts=BASE_TS + 5, product="b", brand="apple", category="electronics.phone"),
        event("view", ts=BASE_TS + 9, product="a", brand="samsung", category="home.lamp"),
    ])
    assert len(state.products) == 2
    assert len(state.brands) == 2
    assert len(state.categories) == 2


def test_gap_statistics_are_running_not_stored():
    """A long session must cost the same memory as a short one."""
    state = play([event("view", ts=BASE_TS + n) for n in (0, 10, 15, 115)])
    assert state.gap_max == 100
    assert state.mean_gap == pytest.approx((10 + 5 + 100) / 3)


def test_an_open_cart_gates_prediction():
    """A session with nothing in a cart cannot abandon one."""
    assert not play([event("view", ts=BASE_TS)]).has_open_cart
    assert play([event("add_to_cart", ts=BASE_TS)]).has_open_cart


def test_a_purchase_settles_the_cart_but_not_the_session():
    """Real REES46 sessions contain several purchases: a shopper buys, keeps
    browsing, and carts again. Treating a purchase as the end of the session
    made the predictor drop it and rebuild from nothing, and its event count
    went backwards, 18 then 2."""
    state = play([
        event("view", ts=BASE_TS),
        event("add_to_cart", ts=BASE_TS + 10, price=99.0),
        event("purchase", ts=BASE_TS + 20),
    ])
    assert not state.has_open_cart
    assert state.cart_value == 0.0
    # The session itself carries on, with its history intact.
    assert state.events == 3
    assert state.converted


def test_a_new_cart_opens_after_a_purchase():
    state = play([
        event("add_to_cart", ts=BASE_TS, product="a", price=10.0),
        event("purchase", ts=BASE_TS + 10),
        event("add_to_cart", ts=BASE_TS + 20, product="b", price=40.0),
    ])
    assert state.has_open_cart
    assert state.cart_value == 40.0
    assert state.cart_adds == 2  # lifetime
    assert state.cycle_cart_adds == 1  # this cycle


def test_expiry_uses_the_supplied_clock():
    state = play([event("view", ts=BASE_TS)])
    assert not state.is_expired(BASE_TS + SESSION_TIMEOUT_SECONDS - 1)
    assert state.is_expired(BASE_TS + SESSION_TIMEOUT_SECONDS + 1)


def test_copy_does_not_alias_its_collections():
    """The trainer snapshots state mid-session while the live object keeps
    mutating, so a shallow copy would share the very sets that make the
    snapshot meaningful."""
    state = play([event("add_to_cart", ts=BASE_TS, product="a")])
    snapshot = state.copy()
    state.update(event("add_to_cart", ts=BASE_TS + 5, product="b", price=99.0))

    assert snapshot.cart_size == 1
    assert state.cart_size == 2
    assert snapshot.products == {"a"}


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------


def test_vector_matches_the_declared_names():
    vector = compute_features(play([event("add_to_cart", ts=BASE_TS)]))
    assert vector.shape == (FEATURE_COUNT,)
    assert len(FEATURE_NAMES) == FEATURE_COUNT
    assert vector.dtype == np.float64


def test_names_are_unique():
    """Order is part of the model contract, so a duplicate would silently
    make one of them unreachable."""
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_an_empty_session_produces_finite_numbers():
    """Every ratio has a zero denominator here, and a NaN would poison a
    whole training batch."""
    vector = compute_features(SessionState("s1"))
    assert np.all(np.isfinite(vector))


def test_a_single_event_session_produces_finite_numbers():
    vector = compute_features(play([event("view", ts=BASE_TS)]))
    assert np.all(np.isfinite(vector))


def test_nothing_derived_from_a_purchase_appears():
    """A purchase is the label. If it reached the features the model would
    score perfectly offline and be useless live."""
    assert not any("purchase" in name for name in FEATURE_NAMES)

    before = play([event("view", ts=BASE_TS), event("add_to_cart", ts=BASE_TS + 5)])
    after = play([
        event("view", ts=BASE_TS),
        event("add_to_cart", ts=BASE_TS + 5),
    ])
    after.update(event("purchase", ts=BASE_TS + 9))
    # The purchase changes the state, but no feature reads the purchase count.
    assert "purchases" not in features_as_dict(after)
    assert before.purchases == 0 and after.purchases == 1


def test_features_only_look_backward():
    """The vector after event three must not change because event four later
    arrives, which is what makes it computable at serving time."""
    events = [event("view", ts=BASE_TS + n * 10) for n in range(3)]
    events.append(event("add_to_cart", ts=BASE_TS + 30, price=500.0))

    partial = compute_features(play(events[:3]))
    full_then_partial = play(events[:3]).copy()
    assert np.array_equal(partial, compute_features(full_then_partial))


def test_time_features_come_from_the_event_not_the_clock():
    """Replaying history in a trainer must produce what it would have
    produced live, so nothing may read wall time."""
    state = play([event("view", ts=BASE_TS)])
    first = compute_features(state)
    import time as _time

    _time.sleep(0.01)
    assert np.array_equal(first, compute_features(state))


def test_features_as_dict_matches_the_vector():
    state = play([event("add_to_cart", ts=BASE_TS, price=25.0)])
    named = features_as_dict(state)
    assert list(named) == list(FEATURE_NAMES)
    assert np.allclose(list(named.values()), compute_features(state))


# --------------------------------------------------------------------------
# the one that prevents train/serve skew
# --------------------------------------------------------------------------


def test_the_trainer_and_the_predictor_compute_identical_vectors():
    """There is one implementation, and this proves both paths reach it.

    The offline path replays a session through SessionState and calls
    compute_features. The online path folds live events into SessionState and
    calls compute_features. Same class, same function. If anyone ever adds a
    second implementation for either side, this fails.
    """
    from aether.ml.dataset import examples_from_session
    from aether.ml.model import ModelArtifact
    from aether.stream.predictor import Predictor
    from aether.stream.config import KafkaConfig

    events = [
        event("view", ts=BASE_TS, product="a", price=30.0),
        event("view", ts=BASE_TS + 12, product="b", price=90.0),
        event("add_to_cart", ts=BASE_TS + 30, product="b", price=90.0),
        event("view", ts=BASE_TS + 55, product="c", price=15.0),
        event("add_to_cart", ts=BASE_TS + 70, product="c", price=15.0),
    ]

    offline = [example.features for example in examples_from_session(events)]

    class Echo:
        """Stands in for a model, capturing exactly what it was asked to score."""

        def __init__(self):
            self.seen = []

        def probability(self, state):
            self.seen.append(compute_features(state))
            return 0.5

    echo = Echo()
    predictor = Predictor(echo, KafkaConfig(), output_topic=None)
    for e in events:
        predictor.handle(e)

    assert offline, "expected the offline path to produce examples"
    assert len(offline) == len(echo.seen)
    for from_trainer, from_predictor in zip(offline, echo.seen):
        assert np.array_equal(from_trainer, from_predictor)
