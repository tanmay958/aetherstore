"""Indexing one Kafka partition into segments.

This is the whole indexer minus Kafka. It knows only that offsets are
monotonic integers, which makes every interesting property, idempotence,
crash recovery, commit ordering, testable without a broker running.

## Why the segment key is what it is

    segments/p0/00000000000000000100-00000000000000000199.seg
                ^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^
                first offset         last offset

Kafka gives at-least-once delivery, not exactly-once, and exactly-once
delivery is not merely hard but impossible over a network that can drop the
acknowledgement of a completed write. What is possible is at-least-once
delivery plus idempotent work, which produces the same *effect*.

Deriving the key from the offset range is what makes the work idempotent. A
crash and replay rewrites a byte-identical object at the same key. Had the key
been a uuid or a timestamp, every retry would have added a second copy of the
same documents to the index, every one of them counted twice by BM25, and the
relevance would have been quietly and permanently wrong with no error
anywhere.

## Why the flush boundary must be deterministic

`max_docs` alone is reproducible: replaying offsets 100 onward always produces
the same boundary and therefore the same key. Adding a timer is not. A first
attempt might flush at offset 150 and crash before committing; the replay,
running on a less busy machine, might reach 199 before the timer fires and
write a different key. Nothing is lost or duplicated, because only the second
is in the manifest, but the first is an orphan consuming storage forever.

So `max_idle_seconds` is off by default, and turning it on means accepting
orphans and eventually needing a collector that deletes objects no manifest
references. Real systems do both.

## Why the manifest is held in memory

Kafka assigns each partition to exactly one consumer in a group, so this
object is the only writer of its manifest. Single-writer safety is inherited
from the consumer group rather than built with a lock or a consensus
algorithm. That is also why the manifest is per partition rather than shared:
one shared file would have several writers and lose segments to overwrites.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from aether.index.manifest import Manifest, SegmentMeta, read_manifest, write_manifest
from aether.index.memory import MemoryIndex
from aether.index.segment import SegmentReader, write_segment
from aether.storage.base import ObjectStore


@dataclass(frozen=True)
class FlushPolicy:
    """When to seal a segment.

    `max_docs` is the deterministic trigger and the only one enabled by
    default. `max_idle_seconds` bounds how long a document can sit unsearchable
    on a quiet topic, at the price of non-reproducible segment boundaries.
    """

    max_docs: int = 10_000
    max_idle_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_docs < 1:
            raise ValueError("max_docs must be at least 1")
        if self.max_idle_seconds is not None and self.max_idle_seconds <= 0:
            raise ValueError("max_idle_seconds must be positive when set")

    @property
    def deterministic(self) -> bool:
        """Whether replaying the same offsets reproduces the same segments."""
        return self.max_idle_seconds is None


def segment_key(partition: int, first_offset: int, last_offset: int) -> str:
    """Fixed width so keys sort in offset order, and derived only from the
    offsets so a replay reproduces it exactly."""
    return f"segments/p{partition}/{first_offset:020d}-{last_offset:020d}.seg"


def manifest_key(partition: int) -> str:
    return f"manifests/p{partition}/current.json"


class PartitionIndexer:
    """Accumulates documents from one partition and seals them into segments."""

    def __init__(
        self,
        store: ObjectStore,
        partition: int,
        *,
        policy: FlushPolicy | None = None,
        clock=time.monotonic,
    ) -> None:
        self.store = store
        self.partition = partition
        self.policy = policy or FlushPolicy()
        self._clock = clock

        self._index = MemoryIndex()
        self._first_offset: int | None = None
        self._last_offset: int | None = None
        self._last_flush_at = clock()
        # The highest offset whose documents are durable and named in the
        # manifest. Everything after it is still only in memory.
        self._committed_through: int | None = None

        # Read once. This consumer owns the partition, so nothing else writes
        # this file and re-reading it would only cost requests.
        self._manifest: Manifest = read_manifest(store, manifest_key(partition))

    # -- state -------------------------------------------------------------

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    @property
    def buffered(self) -> int:
        return self._index.num_docs

    @property
    def next_offset(self) -> int | None:
        """The offset to commit: one past the last one made durable.

        None until something has been flushed, because committing an offset
        for work that is not yet in the manifest would lose it on a crash.
        """
        return None if self._committed_through is None else self._committed_through + 1

    # -- accumulating ------------------------------------------------------

    def add(self, doc: dict, offset: int) -> None:
        """Buffer one document. Nothing is durable until `flush`."""
        if self._first_offset is None:
            self._first_offset = offset
        self._last_offset = offset
        self._index.add(doc)

    def should_flush(self) -> bool:
        if self._index.num_docs == 0:
            return False
        if self._index.num_docs >= self.policy.max_docs:
            return True
        if self.policy.max_idle_seconds is None:
            return False
        return self._clock() - self._last_flush_at >= self.policy.max_idle_seconds

    def discard(self) -> None:
        """Throw away buffered work without writing it.

        What to do when a partition is revoked during a rebalance. The offsets
        were never committed, so whoever inherits the partition replays them.
        Flushing here instead would race the new owner, and both would be
        writing the same manifest.
        """
        self._index = MemoryIndex()
        self._first_offset = None
        self._last_offset = None

    # -- committing --------------------------------------------------------

    def flush(self) -> SegmentMeta | None:
        """Seal the buffer into a segment and publish it.

        Ordering is the point. The segment is written first, then the manifest
        that makes it visible, and only then may the caller commit the offset.
        A crash after the segment leaves invisible garbage that the replay
        overwrites; a crash after the manifest leaves correct data that the
        replay rewrites identically. Committing the offset first would instead
        lose the batch outright.
        """
        if self._index.num_docs == 0:
            return None

        assert self._first_offset is not None and self._last_offset is not None
        key = segment_key(self.partition, self._first_offset, self._last_offset)

        data = write_segment(self._index)
        self.store.put(key, data)

        footer = SegmentReader(self.store, key).footer
        meta = SegmentMeta(
            key,
            footer.num_docs,
            len(data),
            footer.min_ts,
            footer.max_ts,
            first_offset=self._first_offset,
            last_offset=self._last_offset,
        )

        # publish(), not append. A replay can seal a wider offset range than
        # the attempt it is redoing, and listing both would index the
        # overlapping documents twice.
        self._manifest = self._manifest.publish(meta)
        write_manifest(self.store, self._manifest, manifest_key(self.partition))

        self._committed_through = self._last_offset
        self._index = MemoryIndex()
        self._first_offset = None
        self._last_offset = None
        self._last_flush_at = self._clock()
        return meta
