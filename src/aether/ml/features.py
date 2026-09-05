"""Turning session state into a feature vector.

This is one function, `compute_features`, and it is deliberately the only one.

Train/serve skew is the most common way a model that looked good offline
performs badly in production, and it almost always comes from the same cause:
features computed one way in a notebook over a dataframe, and another way in
the serving path over a live request. The two implementations start identical
and drift, usually invisibly, and the model quietly degrades against inputs it
was never trained on.

There is nothing here to drift. The trainer replays events through
`SessionState` and calls this; the predictor folds live events into
`SessionState` and calls this. Same class, same function, same numbers. A test
asserts the two paths produce byte-identical vectors for the same event
sequence, and it would fail the moment anyone added a second implementation.

The features themselves are all *backward looking*. Every one is computable
from events that have already happened, because at serving time the future
does not exist. Nothing derived from a purchase appears here, since a purchase
is the label.
"""

from __future__ import annotations

import numpy as np

from aether.ml.session import SessionState

# Order matters and is part of the model artifact: a model trained with these
# in this order will read garbage if they are ever reordered. The names are
# saved alongside the model and checked at load time.
FEATURE_NAMES: tuple[str, ...] = (
    # How much has happened
    "events",
    "views",
    "clicks",
    "searches",
    "cart_adds",
    "cart_removes",
    # The cart itself, which is what is being abandoned
    "cart_size",
    "cart_value",
    "log_cart_value",
    # Breadth of browsing: someone comparing twenty products behaves
    # differently from someone who went straight to one
    "distinct_products",
    "distinct_categories",
    "distinct_brands",
    # Money. Expensive carts get abandoned more, which is the single most
    # intuitive signal in the problem.
    "mean_price",
    "max_price",
    "log_max_price",
    # Time. Duration and pace separate a decisive buyer from a browser, and a
    # long silence is the strongest late signal that someone has left.
    "duration",
    "log_duration",
    "mean_gap",
    "max_gap",
    "events_per_minute",
    # Ratios, which normalise away session length so a long session and a
    # short one with the same shape look alike
    "view_ratio",
    "cart_add_ratio",
    "remove_ratio",
    "cart_to_view_ratio",
    # Context
    "hour_of_day",
    "is_weekend",
    "is_mobile",
)

FEATURE_COUNT = len(FEATURE_NAMES)

SECONDS_PER_DAY = 86400
SECONDS_PER_HOUR = 3600


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_features(state: SessionState) -> np.ndarray:
    """The feature vector for a session, as of its most recent event.

    Pure, backward-looking, and the only implementation. Returns float64 in
    `FEATURE_NAMES` order.
    """
    events = state.events
    duration = state.duration
    cart_value = state.cart_value

    # Day of week and hour come from the event timestamp rather than from the
    # clock, so replaying history in a trainer produces the same values it
    # would have produced live.
    day_seconds = state.last_ts % SECONDS_PER_DAY
    hour = day_seconds / SECONDS_PER_HOUR
    day_index = (state.last_ts // SECONDS_PER_DAY + 4) % 7  # epoch day 0 was a Thursday

    return np.array(
        [
            events,
            state.views,
            state.clicks,
            state.searches,
            state.cart_adds,
            state.cart_removes,
            state.cart_size,
            cart_value,
            # Prices are heavily right-skewed, and a log gives a tree far more
            # useful split points across the cheap end where most carts live.
            np.log1p(cart_value),
            len(state.products),
            len(state.categories),
            len(state.brands),
            state.mean_price,
            state.price_max,
            np.log1p(state.price_max),
            duration,
            np.log1p(duration),
            state.mean_gap,
            state.gap_max,
            _safe_ratio(events * 60.0, duration),
            _safe_ratio(state.views, events),
            _safe_ratio(state.cart_adds, events),
            _safe_ratio(state.cart_removes, max(state.cart_adds, 1)),
            _safe_ratio(state.cart_adds, max(state.views, 1)),
            hour,
            1.0 if day_index >= 5 else 0.0,
            1.0 if (state.device or "").lower() == "mobile" else 0.0,
        ],
        dtype=np.float64,
    )


def features_as_dict(state: SessionState) -> dict[str, float]:
    """The same vector, named. For dashboards and for explaining a score."""
    return dict(zip(FEATURE_NAMES, compute_features(state).tolist()))
