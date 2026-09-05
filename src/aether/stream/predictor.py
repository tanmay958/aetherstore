"""The predictor service: streaming cart-abandonment inference.

This is the part of the project that is actually about running predictions on
a distributed system, and the whole argument fits in three lines of
configuration.

Events are produced keyed by `session_id`, so Kafka routes every event of a
session to one partition. Kafka gives each partition to exactly one consumer
in a group. Therefore one replica sees a session's entire history, and can
hold its state in a plain Python dict.

    producer:  key = session_id
    kafka:     partition = hash(key) % 12
    group:     one consumer per partition

    predictor #1  ->  partitions 0-3   ->  sessions in a local dict
    predictor #2  ->  partitions 4-7   ->  sessions in a local dict
    predictor #3  ->  partitions 8-11  ->  sessions in a local dict

No shared store, no Redis, no distributed lock, no coordination between
replicas at all. Adding a fourth replica repartitions automatically and each
one still owns whole sessions. Take the key away and the whole thing collapses:
events round-robin across partitions, every replica sees a fragment of every
session, and the state has to move to something shared and locked.

## Different durability requirements to the indexer

The indexer commits Kafka offsets by hand, after the manifest write, because a
duplicated batch would index documents twice and silently corrupt BM25. The
predictor has no such problem: recomputing a prediction produces the same
prediction. Duplicates are harmless, so offsets are committed automatically
and the code is simpler. Requirements differ, so the mechanism differs, and
matching the indexer here out of consistency would be cargo culting.

## Bounded memory

Session state is held for every session currently in flight, which on a busy
stream is a lot of them, and nothing removes a session that simply stopped.
Expired sessions are swept periodically, and without that this process is an
out-of-memory error with a timer on it.

## Rebalancing

A revoked partition's sessions are dropped rather than kept. Whoever inherits
the partition rebuilds state from its own reading of the log, and holding on
would mean two replicas predicting for the same session from different halves
of its history.

    python -m aether.stream.predictor --model data/model.pkl
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

from aether.env import load_dotenv
from aether.ml.model import ModelArtifact
from aether.ml.session import SESSION_TIMEOUT_SECONDS, SessionState
from aether.stream.config import KafkaConfig

DEFAULT_GROUP = "aether-predictors"
DEFAULT_OUTPUT_TOPIC = "predictions"

# How often to sweep expired sessions out of memory, in events rather than
# seconds so a quiet stream does not sweep constantly.
SWEEP_EVERY = 5_000

# Sessions are only scored once they could actually abandon something.
MIN_EVENTS_TO_SCORE = 2


@dataclass
class Prediction:
    session_id: str
    user_id: str
    ts: int
    events: int
    cart_size: int
    cart_value: float
    probability: float
    partition: int

    def to_json(self) -> str:
        return json.dumps(self.__dict__, separators=(",", ":"))


@dataclass
class PredictorStats:
    consumed: int = 0
    scored: int = 0
    sessions_open: int = 0
    sessions_expired: int = 0
    high_risk: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def __str__(self) -> str:
        rate = self.consumed / self.elapsed if self.elapsed else 0
        return (
            f"{self.consumed:,} consumed, {self.scored:,} scored, "
            f"{self.sessions_open:,} sessions open, "
            f"{self.sessions_expired:,} expired, {rate:,.0f}/sec"
        )


class Predictor:
    """One replica: owns some partitions, and the sessions inside them."""

    def __init__(
        self,
        model: ModelArtifact,
        config: KafkaConfig,
        *,
        output_topic: str | None = DEFAULT_OUTPUT_TOPIC,
        risk_threshold: float = 0.7,
        session_timeout: int = SESSION_TIMEOUT_SECONDS,
    ) -> None:
        self.model = model
        self.config = config
        self.output_topic = output_topic
        self.risk_threshold = risk_threshold
        self.session_timeout = session_timeout

        # session_id -> state. Local to this replica, and correct only because
        # partitioning guarantees no other replica sees these sessions.
        self.sessions: dict[str, SessionState] = {}
        # Which partition each session arrived on, kept beside the state
        # rather than inside it because a partition is a fact about Kafka and
        # SessionState is also used by the offline trainer, where it means
        # nothing. Needed so a rebalance can drop exactly the sessions whose
        # partition moved.
        self.partitions: dict[str, int] = {}
        self.stats = PredictorStats()
        self._running = False
        # Newest event timestamp seen, per partition, rather than one figure
        # across all of them. Partitions advance through the log
        # independently: a replica that has just been handed a partition may
        # be reading hour-old events on it while its other partitions are
        # already caught up. A single global watermark takes the maximum
        # across all of them, which made every session on the lagging
        # partition look instantly expired, swept it, and recreated it on the
        # next event. Observed live as a session's event count going
        # backwards, 13 then 4 then 2.
        self._watermarks: dict[int, int] = {}

    # -- the work ----------------------------------------------------------

    def handle(self, event: dict, partition: int = 0) -> Prediction | None:
        """Fold one event into its session and score it.

        Returns a prediction, or None when the session cannot yet abandon
        anything worth predicting about.
        """
        session_id = event["session_id"]
        state = self.sessions.get(session_id)
        if state is None:
            state = self.sessions[session_id] = SessionState(session_id)
        self.partitions[session_id] = partition

        state.update(event)
        self.stats.consumed += 1
        self._watermarks[partition] = max(
            self._watermarks.get(partition, 0), event["ts"]
        )

        # A purchase settles the open cart but does not end the session: real
        # shoppers buy, keep browsing, and cart again. Dropping the session
        # here made its event count go backwards, 18 then 2, when the next
        # event rebuilt it from nothing. The state stays; `has_open_cart`
        # simply goes false until a new cart is started.
        if not state.has_open_cart or state.events < MIN_EVENTS_TO_SCORE:
            return None

        probability = self.model.probability(state)
        self.stats.scored += 1
        if probability >= self.risk_threshold:
            self.stats.high_risk += 1

        return Prediction(
            session_id=session_id,
            user_id=state.user_id,
            ts=state.last_ts,
            events=state.events,
            cart_size=state.cart_size,
            cart_value=round(state.cart_value, 2),
            probability=round(probability, 4),
            partition=partition,
        )

    def sweep(self) -> int:
        """Drop sessions that have gone quiet.

        Uses the newest event timestamp rather than wall clock, so replaying
        history expires sessions the same way live traffic would.
        """
        stale = [
            session_id
            for session_id, state in self.sessions.items()
            if state.is_expired(
                # Compared against its own partition's progress, so a session
                # is never expired by time passing somewhere else.
                self._watermarks.get(self.partitions.get(session_id, -1), state.last_ts),
                self.session_timeout,
            )
        ]
        for session_id in stale:
            del self.sessions[session_id]
            self.partitions.pop(session_id, None)
        self.stats.sessions_expired += len(stale)
        self.stats.sessions_open = len(self.sessions)
        return len(stale)

    def watermark(self, partition: int) -> int:
        """Newest event timestamp seen on one partition."""
        return self._watermarks.get(partition, 0)

    def drop_partitions(self, partitions: set[int]) -> int:
        """Forget every session that arrived on a revoked partition.

        Whoever inherits the partition rebuilds this state from its own
        reading of the log. Keeping it would leave two replicas predicting for
        one session from different halves of its history.
        """
        owned = [
            session_id
            for session_id, partition in self.partitions.items()
            if partition in partitions
        ]
        for session_id in owned:
            self.sessions.pop(session_id, None)
            self.partitions.pop(session_id, None)
        for partition in partitions:
            self._watermarks.pop(partition, None)
        return len(owned)

    # -- the loop ----------------------------------------------------------

    def run(
        self,
        *,
        max_events: int | None = None,
        idle_timeout: float | None = None,
        sink=None,
    ) -> PredictorStats:
        from confluent_kafka import Consumer, KafkaException, Producer

        settings = self.config.consumer_settings()
        settings["group.id"] = self.config.group_id
        # Automatic, unlike the indexer. A duplicated prediction is the same
        # prediction; a duplicated batch of documents corrupts an index.
        settings["enable.auto.commit"] = True

        consumer = Consumer(settings)
        producer = (
            Producer(self.config.producer_settings()) if self.output_topic else None
        )

        def on_revoke(_consumer, revoked) -> None:
            # State for these partitions belongs to whoever inherits them.
            partitions = {tp.partition for tp in revoked}
            dropped = self.drop_partitions(partitions)
            print(f"  revoked {sorted(partitions)}, dropped {dropped} sessions")

        consumer.subscribe([self.config.topic], on_revoke=on_revoke)
        self._running = True
        idle_since = time.monotonic()

        try:
            while self._running:
                message = consumer.poll(1.0)
                if message is None:
                    if idle_timeout and time.monotonic() - idle_since > idle_timeout:
                        print(f"  idle for {idle_timeout}s, stopping")
                        break
                    continue
                if message.error():
                    raise KafkaException(message.error())

                idle_since = time.monotonic()
                event = json.loads(message.value())
                prediction = self.handle(event, message.partition())

                if prediction is not None:
                    if producer is not None:
                        producer.produce(
                            self.output_topic,
                            key=prediction.session_id.encode(),
                            value=prediction.to_json().encode(),
                        )
                        producer.poll(0)
                    if sink is not None:
                        sink(prediction)

                if self.stats.consumed % SWEEP_EVERY == 0:
                    self.sweep()
                if max_events and self.stats.consumed >= max_events:
                    break
        finally:
            self.sweep()
            if producer is not None:
                producer.flush()
            consumer.close()
        return self.stats

    def stop(self) -> None:
        self._running = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aether.stream.predictor",
        description="Score live sessions for cart abandonment.",
    )
    parser.add_argument("--model", type=Path, default=Path("data/model.pkl"))
    parser.add_argument("--group", default=DEFAULT_GROUP, help="consumer group")
    parser.add_argument("--output-topic", default=DEFAULT_OUTPUT_TOPIC)
    parser.add_argument("--no-output", action="store_true", help="do not publish")
    parser.add_argument("--log", type=Path, default=None, help="also append JSONL here")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--idle-timeout", type=float, default=None)
    args = parser.parse_args(argv)
    load_dotenv()

    if not args.model.exists():
        parser.error(f"{args.model} not found. Train one with: python -m aether.ml.train")

    artifact = ModelArtifact.load(args.model)
    base = KafkaConfig.from_env()
    config = KafkaConfig(base.bootstrap_servers, base.topic, args.group)

    predictor = Predictor(
        artifact,
        config,
        output_topic=None if args.no_output else args.output_topic,
        risk_threshold=args.threshold,
    )

    def shutdown(_signum, _frame):
        print("\n  stopping")
        predictor.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    handle = args.log.open("a") if args.log else None
    sink = (lambda p: handle.write(p.to_json() + "\n")) if handle else None

    pr = artifact.metrics.get("model", {}).get("pr_auc")
    print(f"predictor  group={config.group_id}  topic={config.topic}")
    print(f"  model trained on {artifact.trained_on_events:,} events"
          + (f", PR-AUC {pr:.4f}" if pr else ""))
    print(f"  scoring sessions with a cart, flagging above {args.threshold}")

    try:
        stats = predictor.run(
            max_events=args.max_events, idle_timeout=args.idle_timeout, sink=sink
        )
    finally:
        if handle:
            handle.close()

    print()
    print(f"  {stats}")
    print(f"  flagged at risk  {stats.high_risk:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
