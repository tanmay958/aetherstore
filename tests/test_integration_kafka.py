"""End to end against a real broker.

Skipped unless Redpanda is reachable, so the default suite stays offline. The
correctness properties, idempotence, commit ordering, crash recovery, are all
proven in test_partition_indexer.py without a broker. What needs a real one is
the part a fake cannot fake: that consumer groups, manual offset commits, and
rebalancing behave the way the design assumes.

    make up
    uv run pytest -m integration
"""

from __future__ import annotations

import json
import socket
import uuid

import pytest

from aether.data.rees46 import iter_events
from aether.index.coordinator import Coordinator
from aether.index.manifest import read_manifest
from aether.storage import LocalStore
from aether.stream.config import KafkaConfig
from aether.stream.indexer import Indexer
from aether.stream.partition import FlushPolicy, manifest_key

pytestmark = pytest.mark.integration

BOOTSTRAP = "localhost:19092"


def _reachable() -> bool:
    try:
        socket.create_connection(("localhost", 19092), timeout=0.4).close()
    except OSError:
        return False
    return True


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _reachable(), reason="Redpanda not running; try `make up`"),
]


@pytest.fixture
def topic() -> str:
    """A fresh topic per test, so runs cannot contaminate each other."""
    return f"itest-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def config(topic) -> KafkaConfig:
    return KafkaConfig(BOOTSTRAP, topic, f"itest-group-{uuid.uuid4().hex[:8]}")


def publish(config: KafkaConfig, events: list[dict]) -> None:
    from confluent_kafka import Producer

    producer = Producer(config.producer_settings())
    for event in events:
        producer.produce(
            config.topic,
            key=event["session_id"].encode(),
            value=json.dumps(event).encode(),
        )
    producer.flush()


def test_events_become_a_searchable_index(config, sample_csv, tmp_path):
    events = list(iter_events(sample_csv))
    publish(config, events)

    store = LocalStore(tmp_path)
    stats = Indexer(store, config, policy=FlushPolicy(max_docs=10)).run(idle_timeout=5)

    assert stats.consumed == len(events)
    assert stats.documents == len(events)

    # Every partition writes its own manifest, so the coordinator has to be
    # pointed at one. Search across all of them by merging the segment lists.
    total = 0
    for partition in range(12):
        key = manifest_key(partition)
        if store.exists(key):
            total += read_manifest(store, key).docs
    assert total == len(events)


def test_a_restart_resumes_from_the_committed_offset(config, sample_csv, tmp_path):
    """The bookmark is stored in Kafka precisely so it survives the process.
    A restart must not reprocess what was already committed."""
    events = list(iter_events(sample_csv))
    publish(config, events)

    store = LocalStore(tmp_path)
    first = Indexer(store, config, policy=FlushPolicy(max_docs=10)).run(idle_timeout=5)
    assert first.consumed == len(events)

    second = Indexer(store, config, policy=FlushPolicy(max_docs=10)).run(idle_timeout=3)
    assert second.consumed == 0


def test_events_for_a_session_land_in_one_partition(config, sample_csv):
    """Keying by session_id is what lets a consumer own whole sessions, and
    therefore keep session state in a plain dict with no shared store. Without
    a key Kafka round-robins and both properties are lost."""
    from confluent_kafka import Consumer

    events = list(iter_events(sample_csv))
    publish(config, events)

    consumer = Consumer(config.consumer_settings())
    consumer.subscribe([config.topic])
    seen: dict[str, set[int]] = {}
    empty_polls = 0
    while empty_polls < 5:
        message = consumer.poll(1.0)
        if message is None:
            empty_polls += 1
            continue
        assert not message.error()
        session = json.loads(message.value())["session_id"]
        seen.setdefault(session, set()).add(message.partition())
    consumer.close()

    assert seen
    assert all(len(partitions) == 1 for partitions in seen.values())


def test_searching_the_result_matches_the_source(config, sample_csv, tmp_path):
    events = list(iter_events(sample_csv))
    publish(config, events)

    store = LocalStore(tmp_path)
    Indexer(store, config, policy=FlushPolicy(max_docs=100)).run(idle_timeout=5)

    found = 0
    for partition in range(12):
        key = manifest_key(partition)
        if not store.exists(key):
            continue
        coordinator = Coordinator(store, manifest_key=key)
        found += coordinator.search("samsung smartphone", top_k=100).total

    expected = sum(
        1
        for event in events
        if "samsung" in json.dumps(event).lower()
        and "smartphone" in json.dumps(event).lower()
    )
    assert found == expected
