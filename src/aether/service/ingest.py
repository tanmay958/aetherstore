"""Indexing events that arrive over HTTP.

The engine's normal writer is a Kafka consumer, and the safety argument there
is entirely inherited: exactly one consumer owns a partition, so exactly one
process writes that partition's manifest, and nothing needs a lock or a
consensus protocol. None of the code below knows about Kafka. `PartitionIndexer`
only ever needed documents and monotonic offsets, and Kafka was one way to
supply them.

## Where single-writer safety comes from here

There is no broker in front of this, so the property has to come from
somewhere else, and it comes from the deployment: the Cloud Run service runs
with `--max-instances=1`. One instance, one writer, one manifest. That is the
same argument in a different accent, and it is deliberate rather than a
limitation nobody noticed.

It does mean the write path caps read scaling, which is a real cost and the
honest way out is to split them: a second service with `--max-instances=1`
owning ingestion, and the reader scaling freely. That is worth doing when
there is traffic to justify it, and pretending otherwise now would mean
building coordination this project has spent its whole life avoiding.

Within the instance, a lock serialises concurrent requests. That is not
coordination between machines; it is one process not racing itself.

## A dedicated partition

Hosted ingest writes to a partition number outside the range any Kafka
consumer will use, so its manifest sits beside theirs and the coordinator's
union picks it up with no special case. Offsets continue from whatever the
manifest already names rather than restarting at zero: `Manifest.publish`
evicts by offset range, so a writer that restarted its numbering would evict
its own history.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from aether.events import EVENT_FIELDS
from aether.index.manifest import SegmentMeta
from aether.storage.base import ObjectStore
from aether.stream.partition import FlushPolicy, PartitionIndexer, manifest_key

# Outside the range a Kafka topic's partitions occupy, so a streamed index and
# a hosted-ingest index can share a bucket without colliding.
LIVE_PARTITION = 900

# Documents per segment. Small, because an event pushed from a browser should
# be searchable in the next breath rather than when a 10,000-document buffer
# happens to fill. Compaction is what puts the size back.
LIVE_SEGMENT_DOCS = 200


@dataclass(frozen=True)
class Ingested:
    """What one ingest call did."""

    accepted: int
    buffered: int
    flushed: SegmentMeta | None
    searchable: bool
    elapsed_ms: float

    def as_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "buffered": self.buffered,
            "searchable": self.searchable,
            "segment": self.flushed.key if self.flushed else None,
            "documents_in_segment": self.flushed.docs if self.flushed else 0,
            "took_ms": round(self.elapsed_ms, 1),
        }


class LiveIngestor:
    """Turns posted events into segments, and says when they are searchable."""

    def __init__(
        self,
        store: ObjectStore,
        *,
        partition: int = LIVE_PARTITION,
        docs_per_segment: int = LIVE_SEGMENT_DOCS,
    ) -> None:
        self.store = store
        self.partition = partition
        self._lock = threading.Lock()
        self._indexer = PartitionIndexer(
            store, partition, policy=FlushPolicy(max_docs=docs_per_segment)
        )
        # Continue the numbering rather than restarting it. Publish evicts by
        # offset range, so a writer that began again at zero would evict every
        # segment it had already published.
        self._next_offset = 1 + max(
            (
                segment.last_offset
                for segment in self._indexer.manifest.segments
                if segment.last_offset is not None
            ),
            default=-1,
        )

    @property
    def manifest_key(self) -> str:
        return manifest_key(self.partition)

    @property
    def documents(self) -> int:
        return self._indexer.manifest.docs

    def add(self, events: list[dict], *, flush: bool = True) -> Ingested:
        """Index some events.

        Flushes by default, because the caller is a person who has just
        pressed a button and wants to search for what they typed. Batch
        callers pass `flush=False` and let the policy decide, which is what
        keeps segments a sensible size when a stream is running.
        """
        began = time.perf_counter()
        with self._lock:
            for event in events:
                self._indexer.add(_as_document(event), self._next_offset)
                self._next_offset += 1

            sealed = None
            if flush or self._indexer.should_flush():
                sealed = self._indexer.flush()

            return Ingested(
                accepted=len(events),
                buffered=self._indexer.buffered,
                flushed=sealed,
                searchable=sealed is not None,
                elapsed_ms=(time.perf_counter() - began) * 1000,
            )


def _as_document(event: dict) -> dict:
    """Every field the schema names, in order, and nothing else.

    An event that reached here through a JSON body may be missing optional
    fields or carrying extra ones, and a segment's docstore should hold the
    same shape whatever produced it.
    """
    return {field: event.get(field) for field in EVENT_FIELDS}
