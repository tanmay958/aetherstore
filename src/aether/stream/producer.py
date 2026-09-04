"""Feed a REES46 CSV into Kafka.

Stands in for the real click tracker. It streams, so a 5.5 GB month never
lands in memory.

Every record is keyed by `session_id`, and that single line carries most of
the distributed design:

    partition = hash(session_id) % num_partitions

Same key, same partition, always. So every event of a session lands in one
partition, and since Kafka gives each partition to exactly one consumer in a
group, one consumer sees the whole session. The indexer gets single-writer
safety for its manifest out of that, and the predictor later gets to keep
session state in a plain dictionary with no shared store and no locks.

Sending without a key would round-robin records across partitions and destroy
both properties.

    python -m aether.stream.producer data/raw/2019-Oct.csv.gz --limit 100000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from aether.data.rees46 import iter_events
from aether.env import load_dotenv
from aether.stream.config import KafkaConfig


def produce(events, config: KafkaConfig, *, progress_every: int = 50_000) -> int:
    from confluent_kafka import Producer

    producer = Producer(config.producer_settings())
    failures = 0

    def on_delivery(err, _msg):
        nonlocal failures
        if err is not None:
            failures += 1
            print(f"  delivery failed: {err}", file=sys.stderr)

    sent = 0
    for event in events:
        while True:
            try:
                producer.produce(
                    config.topic,
                    key=event["session_id"].encode("utf-8"),
                    value=json.dumps(event, separators=(",", ":")).encode("utf-8"),
                    on_delivery=on_delivery,
                )
                break
            except BufferError:
                # The local queue is full, which means the broker is slower
                # than this loop. Serving delivery callbacks drains it; this
                # is backpressure, and dropping records instead would be a
                # silent data loss bug.
                producer.poll(0.5)
        sent += 1
        if sent % progress_every == 0:
            producer.poll(0)
            print(f"  ... {sent:,} produced", file=sys.stderr)

    producer.flush()
    if failures:
        raise RuntimeError(f"{failures} records failed to deliver")
    return sent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aether.stream.producer",
        description="Stream a REES46 CSV into Kafka, keyed by session.",
    )
    parser.add_argument("input", type=Path, help="REES46 .csv or .csv.gz")
    parser.add_argument("--limit", type=int, default=None, help="stop after N events")
    parser.add_argument("--topic", default=None, help="override the topic")
    args = parser.parse_args(argv)
    load_dotenv()

    if not args.input.exists():
        parser.error(f"{args.input} not found. See docs/DATA.md for how to get it.")

    config = KafkaConfig.from_env()
    if args.topic:
        config = KafkaConfig(config.bootstrap_servers, args.topic, config.group_id)

    print(f"producing {args.input} -> {config.topic} on {config.bootstrap_servers}")
    began = time.perf_counter()
    sent = produce(iter_events(args.input, limit=args.limit), config)
    elapsed = time.perf_counter() - began

    print(f"  produced       {sent:,} events")
    print(f"  elapsed        {elapsed:,.1f} s")
    if elapsed > 0:
        print(f"  throughput     {sent / elapsed:,.0f} events/sec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
