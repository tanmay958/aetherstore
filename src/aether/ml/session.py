"""What we know about a shopping session so far.

One of these per live session, updated event by event. It is the only state
the predictor keeps, and it is deliberately small: everything here is a
counter, a running total, or a set bounded by the number of distinct products
one person looks at in one sitting.

That size matters because of where it lives. Kafka routes every event for a
session to exactly one predictor replica, so this object sits in a plain
Python dict on that replica with no shared store, no Redis, and no locking.
The whole distributed inference design rests on this staying cheap enough to
hold in memory for every session currently in flight.

Two rules keep the state honest.

**Nothing here looks forward.** The state after event five knows about events
one to five and nothing else. That is what makes it usable at serving time,
where the future genuinely does not exist yet, and it is what stops a training
set from leaking outcomes into its own features.

**Purchases are recorded but never featurised.** A purchase is the label. It
is tracked so the trainer can tell which carts converted, and every feature is
computed from state before the purchase that resolves it.

## A purchase ends a cart, not a session

Real REES46 sessions routinely contain several purchases. A shopper buys one
thing, keeps browsing, carts more, and buys again; one observed session has
four purchases across thirty-seven events. Treating a purchase as the end of
the session was wrong in a way that only showed up against real data: the
predictor dropped the session, rebuilt it from the next event, and its event
count went backwards, 18 then 2.

So a purchase clears the cart and opens a new cycle, while the session's
history, its duration, its browsing breadth, its pace, carries on. What can be
abandoned is the cart currently open, and `has_open_cart` says whether there
is one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aether.events import CART_EVENT, PURCHASE_EVENT

# A session is considered over after this long without an event. REES46 rotates
# its own session ids after a long pause, so this mostly matters for deciding
# when a live session can be dropped from memory and labelled.
SESSION_TIMEOUT_SECONDS = 30 * 60


@dataclass
class SessionState:
    """Everything known about one session, as of the last event seen."""

    session_id: str
    user_id: str = ""
    device: str | None = None

    first_ts: int = 0
    last_ts: int = 0

    events: int = 0
    views: int = 0
    clicks: int = 0
    searches: int = 0
    cart_adds: int = 0
    cart_removes: int = 0
    purchases: int = 0
    # Adds since the last purchase. A session with three purchases has had
    # three cart cycles, and only the current one can still be abandoned.
    cycle_cart_adds: int = 0

    # Cart contents, so its value survives a removal correctly rather than
    # being tracked as a running sum that removals can only guess at.
    cart: dict[str, float] = field(default_factory=dict)

    products: set[str] = field(default_factory=set)
    categories: set[str] = field(default_factory=set)
    brands: set[str] = field(default_factory=set)

    price_sum: float = 0.0
    price_count: int = 0
    price_max: float = 0.0

    # Running gap statistics rather than a list of every timestamp, so a long
    # session costs the same memory as a short one.
    gap_sum: int = 0
    gap_count: int = 0
    gap_max: int = 0

    def update(self, event: dict) -> None:
        """Fold one event into the state. Must be called in event order."""
        ts = event["ts"]
        if self.events == 0:
            self.first_ts = ts
            self.user_id = event.get("user_id", "")
            self.device = event.get("device")
        else:
            gap = max(0, ts - self.last_ts)
            self.gap_sum += gap
            self.gap_count += 1
            self.gap_max = max(self.gap_max, gap)

        self.last_ts = ts
        self.events += 1

        event_type = event["event_type"]
        if event_type == "view":
            self.views += 1
        elif event_type == "click":
            self.clicks += 1
        elif event_type == "search":
            self.searches += 1
        elif event_type == CART_EVENT:
            self.cart_adds += 1
        elif event_type == "remove_from_cart":
            self.cart_removes += 1
        elif event_type == PURCHASE_EVENT:
            self.purchases += 1
            # The cart is settled. Everything else about the session stands.
            self.cart.clear()
            self.cycle_cart_adds = 0

        product = event.get("product_id")
        price = event.get("price")
        if product:
            self.products.add(product)
            if event_type == CART_EVENT:
                self.cycle_cart_adds += 1
                if price is not None:
                    self.cart[product] = price
            elif event_type == "remove_from_cart":
                self.cart.pop(product, None)

        if event.get("category"):
            self.categories.add(event["category"])
        if event.get("brand"):
            self.brands.add(event["brand"])

        if price is not None:
            self.price_sum += price
            self.price_count += 1
            self.price_max = max(self.price_max, price)

    # -- derived, and used by features -------------------------------------

    @property
    def duration(self) -> int:
        return self.last_ts - self.first_ts

    @property
    def cart_value(self) -> float:
        return sum(self.cart.values())

    @property
    def cart_size(self) -> int:
        return len(self.cart)

    @property
    def has_open_cart(self) -> bool:
        """Whether there is currently something that could be abandoned.

        False before the first add, and false again after a purchase settles
        the cart, until the shopper starts a new one. A session with nothing
        in a cart is neither a positive nor a negative example, so it is
        excluded from both training and prediction.
        """
        return self.cycle_cart_adds > 0

    @property
    def converted(self) -> bool:
        """Whether this session has ever bought anything.

        Session-level, and deliberately not the label: the label is about the
        cart currently open, and a session that bought once can still abandon
        the next cart it fills.
        """
        return self.purchases > 0

    @property
    def mean_gap(self) -> float:
        return self.gap_sum / self.gap_count if self.gap_count else 0.0

    @property
    def mean_price(self) -> float:
        return self.price_sum / self.price_count if self.price_count else 0.0

    def is_expired(self, now: int, timeout: int = SESSION_TIMEOUT_SECONDS) -> bool:
        return now - self.last_ts > timeout

    def copy(self) -> SessionState:
        """A snapshot, for capturing state mid-session without aliasing.

        The trainer needs the state as it stood at each event, and the live
        object keeps mutating, so a shallow copy would share the very sets and
        dicts that make the snapshot meaningful.
        """
        clone = SessionState(self.session_id)
        clone.__dict__.update(self.__dict__)
        clone.cart = dict(self.cart)
        clone.products = set(self.products)
        clone.categories = set(self.categories)
        clone.brands = set(self.brands)
        return clone
