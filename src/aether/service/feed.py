"""Where the hosted stream gets its events.

The raw REES46 CSV is 14.7 GB and is a build input, never a deployment
artifact, so the service cannot read it. What it reads instead is that file
already turned into canonical events, gzipped, and cut into numbered chunks:

    feed/00000.jsonl.gz
    feed/00001.jsonl.gz

Chunks rather than one object because gzip has to be decompressed from the
start, so seeking into a single stream means paying for everything before the
point of interest. A numbered chunk is one range read of a known key.

## Resuming without keeping state

A fresh instance has no memory of how far a previous one got, and writing a
cursor somewhere would mean another object and another writer.

Instead the position is derived: the live partition's manifest already says
how many documents have been indexed, and at a fixed chunk size that is the
chunk to read next. It is approximate, since a partly consumed chunk rounds
down, so a restart can replay a few events already indexed. Those become
duplicate documents rather than lost ones, which is the right way round for a
demo feed, and the alternative is a second thing to keep consistent.
"""

from __future__ import annotations

import gzip
import io
import json
import threading

from aether.storage.base import ObjectStore

DEFAULT_PREFIX = "feed/"
CHUNK_EVENTS = 2_000


def chunk_key(number: int, prefix: str = DEFAULT_PREFIX) -> str:
    return f"{prefix}{number:05d}.jsonl.gz"


class Feed:
    """A supply of real events, read a chunk at a time."""

    def __init__(
        self,
        store: ObjectStore,
        *,
        prefix: str = DEFAULT_PREFIX,
        chunk_events: int = CHUNK_EVENTS,
        start_after: int = 0,
    ) -> None:
        self.store = store
        self.prefix = prefix
        self.chunk_events = chunk_events
        self._chunk = start_after // chunk_events
        self._buffer: list[dict] = []
        self._exhausted = False
        self._lock = threading.Lock()

    @property
    def exhausted(self) -> bool:
        return self._exhausted and not self._buffer

    def take(self, count: int) -> list[dict]:
        """Up to `count` events, or fewer at the end of the feed."""
        with self._lock:
            while len(self._buffer) < count and not self._exhausted:
                self._load_next_chunk()
            taken, self._buffer = self._buffer[:count], self._buffer[count:]
            return taken

    def _load_next_chunk(self) -> None:
        key = chunk_key(self._chunk, self.prefix)
        try:
            raw = self.store.get_range(key, 0, self.store.size(key))
        except Exception:  # noqa: BLE001 - any miss is the end of the feed
            self._exhausted = True
            return
        self._chunk += 1
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as unzipped:
            self._buffer.extend(
                json.loads(line) for line in unzipped if line.strip()
            )


def write_feed(
    store: ObjectStore,
    events,
    *,
    prefix: str = DEFAULT_PREFIX,
    chunk_events: int = CHUNK_EVENTS,
    limit: int | None = None,
) -> tuple[int, int]:
    """Publish events as numbered chunks. Returns (chunks, events)."""
    chunk: list[dict] = []
    written = total = 0

    def flush() -> None:
        nonlocal chunk, written
        if not chunk:
            return
        payload = io.BytesIO()
        with gzip.GzipFile(fileobj=payload, mode="wb", mtime=0) as out:
            for event in chunk:
                out.write((json.dumps(event, separators=(",", ":")) + "\n").encode())
        store.put(chunk_key(written, prefix), payload.getvalue())
        written += 1
        chunk = []

    for event in events:
        chunk.append(event)
        total += 1
        if len(chunk) >= chunk_events:
            flush()
        if limit and total >= limit:
            break
    flush()
    return written, total
