"""The indexer service: Kafka to segments.

A thin shell around `PartitionIndexer`, which holds all the logic worth
testing. What lives here is the part that genuinely needs a broker: consuming,
rebalancing, and committing offsets in the right order.

## The commit order, and why it is that way

    1. consume offsets 100..199
    2. build the index in memory
    3. PUT segments/p0/...100-199.seg
    4. PUT manifests/p0/current.json     <- the data becomes searchable here
    5. commit offset 200                 <- and only now is it acknowledged

Work first, bookmark last. The alternative, committing first, is at-most-once:
a crash between the two loses the batch permanently with nothing to detect it.
This order is at-least-once, so a crash duplicates instead, and duplication is
survivable because the segment key is derived from the offset range. A replay
rewrites a byte-identical object at the same key, so the duplicate cannot
exist.

Crash after each step:

    after 3   segment exists, unreferenced, invisible. Replay overwrites it.
    after 4   data searchable, offset not committed. Replay rewrites both.
    after 5   nothing to redo.

## Rebalancing

When partitions move between consumers, buffered work for a revoked partition
is thrown away rather than flushed. Its offsets were never committed, so
whoever inherits the partition replays them. Flushing on the way out would
race the new owner, and both would be writing the same manifest.

    python -m aether.stream.indexer r2://aether --max-docs 5000
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass, field

from aether.env import load_dotenv
from aether.storage import open_store
from aether.storage.base import ObjectStore
from aether.stream.config import KafkaConfig
from aether.stream.partition import FlushPolicy, PartitionIndexer


@dataclass
class IndexerStats:
    consumed: int = 0
    segments: int = 0
    documents: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def __str__(self) -> str:
        rate = self.consumed / self.elapsed if self.elapsed else 0
        return (
            f"{self.consumed:,} consumed, {self.segments:,} segments, "
            f"{self.documents:,} indexed, {rate:,.0f}/sec"
        )


class Indexer:
    """Consume a topic and turn it into segments."""

    def __init__(
        self,
        store: ObjectStore,
        config: KafkaConfig,
        *,
        policy: FlushPolicy | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self.policy = policy or FlushPolicy()
        self.stats = IndexerStats()
        self.partitions: dict[int, PartitionIndexer] = {}
        self._running = False

    # -- rebalancing -------------------------------------------------------

    def _on_assign(self, consumer, assigned) -> None:
        for tp in assigned:
            # Reading the manifest happens here, once, because this consumer
            # is now the only writer of it.
            self.partitions[tp.partition] = PartitionIndexer(
                self.store, tp.partition, policy=self.policy
            )
        print(f"  assigned partitions {sorted(tp.partition for tp in assigned)}")

    def _on_revoke(self, consumer, revoked) -> None:
        for tp in revoked:
            indexer = self.partitions.pop(tp.partition, None)
            if indexer and indexer.buffered:
                # Never flush on the way out. These offsets were never
                # committed, so the next owner will replay them; writing now
                # would race that owner on the same manifest.
                print(
                    f"  revoked p{tp.partition}, discarding {indexer.buffered} "
                    "buffered docs for replay"
                )
                indexer.discard()
        print(f"  revoked partitions {sorted(tp.partition for tp in revoked)}")

    # -- the loop ----------------------------------------------------------

    def _commit(self, consumer, partition: int, indexer: PartitionIndexer) -> None:
        from confluent_kafka import TopicPartition

        next_offset = indexer.next_offset
        if next_offset is None:
            return
        consumer.commit(
            offsets=[TopicPartition(self.config.topic, partition, next_offset)],
            asynchronous=False,
        )

    def run(self, *, max_events: int | None = None, idle_timeout: float | None = None):
        from confluent_kafka import Consumer, KafkaException

        consumer = Consumer(self.config.consumer_settings())
        consumer.subscribe(
            [self.config.topic], on_assign=self._on_assign, on_revoke=self._on_revoke
        )
        self._running = True
        idle_since = time.monotonic()

        try:
            while self._running:
                message = consumer.poll(1.0)

                if message is None:
                    # Quiet topic. Flush anything the policy says is overdue,
                    # then decide whether to stop.
                    for partition, indexer in self.partitions.items():
                        if indexer.should_flush():
                            self._record(indexer.flush())
                            self._commit(consumer, partition, indexer)
                    if idle_timeout and time.monotonic() - idle_since > idle_timeout:
                        print(f"  idle for {idle_timeout}s, stopping")
                        break
                    continue

                if message.error():
                    raise KafkaException(message.error())

                idle_since = time.monotonic()
                indexer = self.partitions.get(message.partition())
                if indexer is None:
                    # A message for a partition revoked between poll and now.
                    continue

                indexer.add(json.loads(message.value()), message.offset())
                self.stats.consumed += 1

                if indexer.should_flush():
                    self._record(indexer.flush())
                    self._commit(consumer, message.partition(), indexer)

                if max_events and self.stats.consumed >= max_events:
                    break

            # Seal whatever is left, so a clean shutdown does not strand
            # documents that a crash would simply have replayed.
            for partition, indexer in self.partitions.items():
                if indexer.buffered:
                    self._record(indexer.flush())
                    self._commit(consumer, partition, indexer)
        finally:
            consumer.close()
        return self.stats

    def _record(self, meta) -> None:
        if meta is None:
            return
        self.stats.segments += 1
        self.stats.documents += meta.docs
        print(f"  flushed {meta.key}  {meta.docs:,} docs  {meta.bytes:,} B")

    def stop(self) -> None:
        self._running = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aether.stream.indexer",
        description="Consume a Kafka topic into segments.",
    )
    parser.add_argument("output", help="a directory, s3://bucket/prefix, or r2://...")
    parser.add_argument("--max-docs", type=int, default=10_000, help="documents per segment")
    parser.add_argument(
        "--max-idle",
        type=float,
        default=None,
        help="also flush after N idle seconds. Makes segment boundaries "
        "non-reproducible, so a replay can orphan objects; see partition.py",
    )
    parser.add_argument("--max-events", type=int, default=None, help="stop after N events")
    parser.add_argument(
        "--idle-timeout", type=float, default=None, help="stop after N quiet seconds"
    )
    args = parser.parse_args(argv)
    load_dotenv()

    config = KafkaConfig.from_env()
    indexer = Indexer(
        open_store(args.output),
        config,
        policy=FlushPolicy(max_docs=args.max_docs, max_idle_seconds=args.max_idle),
    )

    def shutdown(_signum, _frame):
        print("\n  shutting down, sealing buffered work")
        indexer.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"indexing {config.topic} on {config.bootstrap_servers} -> {args.output}")
    if not indexer.policy.deterministic:
        print("  note: --max-idle makes segment keys non-reproducible", file=sys.stderr)

    stats = indexer.run(max_events=args.max_events, idle_timeout=args.idle_timeout)
    print()
    print(f"  {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
