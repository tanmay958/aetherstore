"""Kafka connection settings.

Redpanda is used locally rather than Apache Kafka: same wire protocol, one
binary, no ZooKeeper, and around 500MB of RAM. Nothing here is Redpanda
specific, so the same code runs against any Kafka cluster.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_BOOTSTRAP = "localhost:19092"
DEFAULT_TOPIC = "clickstream"
DEFAULT_GROUP = "aether-indexers"


@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: str = DEFAULT_BOOTSTRAP
    topic: str = DEFAULT_TOPIC
    group_id: str = DEFAULT_GROUP

    @classmethod
    def from_env(cls) -> KafkaConfig:
        return cls(
            os.getenv("AETHER_KAFKA_BOOTSTRAP", DEFAULT_BOOTSTRAP),
            os.getenv("AETHER_KAFKA_TOPIC", DEFAULT_TOPIC),
            os.getenv("AETHER_KAFKA_GROUP", DEFAULT_GROUP),
        )

    def consumer_settings(self) -> dict:
        return {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.group_id,
            # Offsets are committed by hand, after the manifest write. Auto
            # commit would move the bookmark on a timer regardless of whether
            # the work was durable, which is precisely the at-most-once
            # failure: a crash in the wrong moment silently loses a batch.
            "enable.auto.commit": False,
            # A new group starts at the beginning of the topic rather than
            # skipping everything written before it existed.
            "auto.offset.reset": "earliest",
            "enable.partition.eof": False,
        }

    def producer_settings(self) -> dict:
        return {
            "bootstrap.servers": self.bootstrap_servers,
            # Wait for every in-sync replica. A producer that does not wait
            # can lose messages on broker failure, and an index cannot recover
            # data that never reached the log.
            "acks": "all",
            # The broker deduplicates producer retries, so a retried send
            # cannot append the same record twice.
            "enable.idempotence": True,
            "linger.ms": 20,
            "compression.type": "lz4",
        }
