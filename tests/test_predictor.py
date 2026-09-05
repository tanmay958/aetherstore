"""Tests for the streaming predictor.

The claim being defended is the distributed one: because the stream is keyed
by session, one replica owns a session's entire history and can keep its state
in a local dict with no shared store and no locking. Most of this file
simulates several replicas and checks that property actually holds, including
across a rebalance, without needing a broker.
"""

import pytest

from aether.ml.features import compute_features
from aether.ml.session import SESSION_TIMEOUT_SECONDS, SessionState
from aether.stream.config import KafkaConfig
from aether.stream.predictor import MIN_EVENTS_TO_SCORE, Predictor

BASE_TS = 1_570_000_000
PARTITIONS = 12


class FixedModel:
    """Returns a set probability, and records what it was asked about."""

    def __init__(self, probability: float = 0.8) -> None:
        self.probability_value = probability
        self.seen: list[SessionState] = []

    def probability(self, state: SessionState) -> float:
        self.seen.append(state.copy())
        return self.probability_value


def event(kind: str, ts: int, session: str = "s1", price: float = 10.0) -> dict:
    return {
        "event_id": f"{session}-{ts}",
        "ts": ts,
        "session_id": session,
        "user_id": f"u-{session}",
        "event_type": kind,
        "device": "mobile",
        "product_id": "p1",
        "title": "thing",
        "category": "electronics.phone",
        "brand": "samsung",
        "price": price,
        "query": None,
    }


def partition_for(session_id: str, partitions: int = PARTITIONS) -> int:
    """Stand-in for Kafka's `hash(key) % partitions`.

    The exact hash does not matter. What matters is that it is a function of
    the key alone, so the same session always lands in the same place.
    """
    import zlib

    return zlib.crc32(session_id.encode()) % partitions


@pytest.fixture
def predictor():
    return Predictor(FixedModel(), KafkaConfig(), output_topic=None)


# --------------------------------------------------------------------------
# when a session gets scored
# --------------------------------------------------------------------------


def test_a_session_without_a_cart_is_not_scored(predictor):
    assert predictor.handle(event("view", BASE_TS)) is None
    assert predictor.handle(event("view", BASE_TS + 5)) is None
    assert predictor.model.seen == []


def test_a_carted_session_is_scored(predictor):
    predictor.handle(event("view", BASE_TS))
    prediction = predictor.handle(event("add_to_cart", BASE_TS + 5))

    assert prediction is not None
    assert prediction.session_id == "s1"
    assert prediction.cart_size == 1
    assert prediction.probability == pytest.approx(0.8)


def test_a_session_is_scored_again_on_every_later_event(predictor):
    predictor.handle(event("view", BASE_TS))
    predictor.handle(event("add_to_cart", BASE_TS + 5))
    predictions = [predictor.handle(event("view", BASE_TS + 10 + n)) for n in range(3)]
    assert all(p is not None for p in predictions)


def test_a_single_event_session_is_not_scored(predictor):
    """One event is not enough to say anything about intent."""
    assert MIN_EVENTS_TO_SCORE > 1
    assert predictor.handle(event("add_to_cart", BASE_TS)) is None


def test_a_purchase_stops_scoring_without_dropping_the_session(predictor):
    """Found against real data. REES46 sessions routinely contain several
    purchases, so dropping the session on one made the predictor rebuild it
    from the next event and its event count went backwards, 18 then 2. The
    purchase settles the cart; the session's history stands."""
    predictor.handle(event("view", BASE_TS))
    predictor.handle(event("add_to_cart", BASE_TS + 5))

    assert predictor.handle(event("purchase", BASE_TS + 9)) is None
    assert "s1" in predictor.sessions
    assert predictor.sessions["s1"].events == 3


def test_a_second_cart_in_one_session_is_scored_again(predictor):
    """And the count keeps going up, rather than restarting."""
    for kind, offset in (("view", 0), ("add_to_cart", 5), ("purchase", 9)):
        predictor.handle(event(kind, BASE_TS + offset))

    prediction = predictor.handle(event("add_to_cart", BASE_TS + 20, price=75.0))

    assert prediction is not None
    assert prediction.events == 4
    assert prediction.cart_value == pytest.approx(75.0)


def test_the_prediction_carries_what_a_dashboard_needs(predictor):
    predictor.handle(event("view", BASE_TS))
    prediction = predictor.handle(event("add_to_cart", BASE_TS + 5, price=249.0))

    assert prediction.cart_value == pytest.approx(249.0)
    assert prediction.events == 2
    assert prediction.user_id == "u-s1"
    assert "session_id" in prediction.to_json()


# --------------------------------------------------------------------------
# state is per session, and local
# --------------------------------------------------------------------------


def test_sessions_do_not_contaminate_each_other(predictor):
    predictor.handle(event("add_to_cart", BASE_TS, "a", price=500.0))
    predictor.handle(event("add_to_cart", BASE_TS + 1, "b", price=5.0))
    a = predictor.handle(event("view", BASE_TS + 2, "a"))
    b = predictor.handle(event("view", BASE_TS + 3, "b"))

    assert a.cart_value == pytest.approx(500.0)
    assert b.cart_value == pytest.approx(5.0)


def test_the_model_sees_the_state_as_it_stood(predictor):
    for n in range(4):
        predictor.handle(event("add_to_cart", BASE_TS + n * 10, price=10.0 * (n + 1)))

    seen = predictor.model.seen
    assert [state.events for state in seen] == [2, 3, 4]


# --------------------------------------------------------------------------
# bounded memory
# --------------------------------------------------------------------------


def test_expired_sessions_are_swept(predictor):
    predictor.handle(event("add_to_cart", BASE_TS, "old"))
    predictor.handle(event("add_to_cart", BASE_TS + SESSION_TIMEOUT_SECONDS * 2, "new"))

    swept = predictor.sweep()

    assert swept == 1
    assert set(predictor.sessions) == {"new"}
    assert predictor.stats.sessions_expired == 1


def test_sweeping_uses_event_time_not_wall_clock(predictor):
    """Replaying history must expire sessions the way live traffic would,
    otherwise a backfill holds every session it ever saw."""
    predictor.handle(event("add_to_cart", BASE_TS, "a"))
    assert predictor.sweep() == 0  # nothing newer has arrived yet

    predictor.handle(event("view", BASE_TS + SESSION_TIMEOUT_SECONDS * 3, "b"))
    assert predictor.sweep() == 1


def test_sweeping_clears_the_partition_map_too(predictor):
    """A leak here would be invisible: predictions stay correct while memory
    grows forever."""
    predictor.handle(event("add_to_cart", BASE_TS, "a"))
    predictor.handle(event("view", BASE_TS + SESSION_TIMEOUT_SECONDS * 3, "b"))
    predictor.sweep()
    assert set(predictor.partitions) == set(predictor.sessions)


# --------------------------------------------------------------------------
# the distributed claim
# --------------------------------------------------------------------------


def test_keying_by_session_sends_every_event_to_one_partition():
    """The property everything else rests on. Without a key Kafka
    round-robins, and each replica would see a fragment of every session."""
    sessions = [f"s{i:04d}" for i in range(500)]
    for session_id in sessions:
        landings = {partition_for(session_id) for _ in range(10)}
        assert len(landings) == 1


def test_three_replicas_each_own_whole_sessions():
    """The JD gap, simulated. Twelve partitions across three replicas, events
    routed by session key, and no replica ever sees a session another one is
    also tracking."""
    replicas = {
        name: Predictor(FixedModel(), KafkaConfig(), output_topic=None)
        for name in ("r1", "r2", "r3")
    }
    assignment = {p: ("r1", "r2", "r3")[p % 3] for p in range(PARTITIONS)}

    events = []
    for i in range(200):
        session_id = f"s{i:04d}"
        for n in range(4):
            kind = "add_to_cart" if n == 1 else "view"
            events.append(event(kind, BASE_TS + i * 5 + n, session_id))

    for e in events:
        partition = partition_for(e["session_id"])
        replicas[assignment[partition]].handle(e, partition)

    # No session is tracked by more than one replica.
    seen: dict[str, str] = {}
    for name, replica in replicas.items():
        for session_id in replica.sessions:
            assert session_id not in seen, f"{session_id} on {seen.get(session_id)} and {name}"
            seen[session_id] = name

    # And between them they saw every session, with nothing lost.
    assert len(seen) == 200
    assert sum(r.stats.consumed for r in replicas.values()) == len(events)


def test_a_replica_sees_a_sessions_whole_history():
    """The reason a local dict is sufficient. If events for one session were
    split across replicas, each would score from half the story."""
    replica = Predictor(FixedModel(), KafkaConfig(), output_topic=None)
    for n in range(6):
        kind = "add_to_cart" if n == 2 else "view"
        replica.handle(event(kind, BASE_TS + n * 10, "s1"), partition_for("s1"))

    assert replica.sessions["s1"].events == 6
    assert replica.model.seen[-1].events == 6


# --------------------------------------------------------------------------
# rebalancing
# --------------------------------------------------------------------------


def test_a_revoked_partition_drops_exactly_its_sessions(predictor):
    predictor.handle(event("add_to_cart", BASE_TS, "a"), partition=3)
    predictor.handle(event("add_to_cart", BASE_TS + 1, "b"), partition=7)

    dropped = predictor.drop_partitions({3})

    assert dropped == 1
    assert set(predictor.sessions) == {"b"}
    assert set(predictor.partitions) == {"b"}


def test_the_new_owner_rebuilds_state_from_the_log():
    """Dropped rather than handed over. Whoever inherits the partition reads
    the same events and arrives at the same state, so holding on would leave
    two replicas predicting for one session from different halves of it."""
    losing = Predictor(FixedModel(), KafkaConfig(), output_topic=None)
    events = [
        event("view", BASE_TS, "s1"),
        event("add_to_cart", BASE_TS + 10, "s1", price=99.0),
        event("view", BASE_TS + 20, "s1"),
    ]
    for e in events:
        losing.handle(e, partition=5)
    before = compute_features(losing.sessions["s1"])

    losing.drop_partitions({5})
    assert not losing.sessions

    inheriting = Predictor(FixedModel(), KafkaConfig(), output_topic=None)
    for e in events:
        inheriting.handle(e, partition=5)

    assert (compute_features(inheriting.sessions["s1"]) == before).all()


def test_revoking_an_unowned_partition_is_harmless(predictor):
    predictor.handle(event("add_to_cart", BASE_TS, "a"), partition=1)
    assert predictor.drop_partitions({9}) == 0
    assert set(predictor.sessions) == {"a"}


def test_a_lagging_partition_does_not_expire_its_own_sessions(predictor):
    """Found by running three replicas against real Kafka.

    Partitions advance through the log independently. A replica handed a new
    partition may be reading hour-old events on it while its other partitions
    are already caught up. With one watermark across all of them, the maximum
    came from the caught-up partitions, so every session on the lagging one
    looked instantly expired, was swept, and was recreated on its next event.
    It showed up as a session's event count going backwards: 13, then 4,
    then 2.
    """
    # Partition 7 is far ahead.
    predictor.handle(event("view", BASE_TS + SESSION_TIMEOUT_SECONDS * 5, "ahead"), 7)
    # Partition 0 has just been assigned and is replaying older events.
    predictor.handle(event("add_to_cart", BASE_TS, "behind"), 0)
    predictor.handle(event("view", BASE_TS + 30, "behind"), 0)

    predictor.sweep()

    assert "behind" in predictor.sessions, "a lagging partition's session was expired"
    assert predictor.sessions["behind"].events == 2


def test_a_session_still_expires_once_its_own_partition_moves_on(predictor):
    predictor.handle(event("add_to_cart", BASE_TS, "old"), 0)
    predictor.handle(event("view", BASE_TS + SESSION_TIMEOUT_SECONDS * 3, "new"), 0)

    predictor.sweep()

    assert set(predictor.sessions) == {"new"}


def test_watermarks_are_tracked_per_partition(predictor):
    predictor.handle(event("view", BASE_TS, "a"), 0)
    predictor.handle(event("view", BASE_TS + 9_999, "b"), 5)

    assert predictor.watermark(0) == BASE_TS
    assert predictor.watermark(5) == BASE_TS + 9_999


def test_a_revoked_partition_forgets_its_watermark(predictor):
    predictor.handle(event("view", BASE_TS, "a"), 3)
    predictor.drop_partitions({3})
    assert predictor.watermark(3) == 0
