"""A store that counts what was asked of it.

On object storage the request count *is* the cost model. A range GET takes
20-50ms and costs money whether it returns 64 bytes or 8 megabytes, so the
number of requests, not the volume of data, is what you pay in both latency
and dollars.

That makes timing alone misleading during development. Every range read here
is a local `seek`, roughly three hundred times faster than a network round
trip, so a query that issues forty pointless requests still looks instant. The
request count is the number that survives the move to S3 unchanged, which is
why it gets measured rather than inferred.

This wraps any store, so it works identically over local files today and over
S3 later:

    store = CountingStore(LocalStore("data"))
    segment = SegmentReader(store, "0.seg")
    with store.measure() as cost:
        segment.search_and("samsung smartphone")
    print(cost)          # 4 requests, 2.1 KB
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from aether.storage.base import ObjectStore


@dataclass
class ReadStats:
    """What a stretch of work cost in storage terms."""

    requests: int = 0
    bytes_read: int = 0

    def __str__(self) -> str:
        if self.bytes_read < 1024:
            size = f"{self.bytes_read} B"
        elif self.bytes_read < 1_048_576:
            size = f"{self.bytes_read / 1024:.1f} KB"
        else:
            size = f"{self.bytes_read / 1_048_576:.2f} MB"
        plural = "" if self.requests == 1 else "s"
        return f"{self.requests} request{plural}, {size}"


@dataclass
class Read:
    """One range read, kept only when tracing is on."""

    key: str
    start: int | None
    length: int
    returned: int


class CountingStore(ObjectStore):
    """Wraps a store and records every read that passes through it.

    Writes are not counted. They matter for cost too, but they happen once per
    segment during indexing rather than on every query, so they are not what
    the query path is being judged on.
    """

    def __init__(self, inner: ObjectStore, *, trace: bool = False) -> None:
        self.inner = inner
        self.stats = ReadStats()
        self.trace = trace
        self.reads: list[Read] = []
        # The coordinator fans out across segments on a thread pool, so
        # several reads land here concurrently. Without this the counts drift
        # low under load, which is the worst possible failure for a number
        # whose entire job is to be trusted.
        self._lock = threading.Lock()

    def reset(self) -> None:
        self.stats = ReadStats()
        self.reads.clear()

    @contextmanager
    def measure(self) -> Iterator[ReadStats]:
        """Cost of one stretch of work, isolated from everything before it.

        Useful for separating the price of opening a segment, which is paid
        once and cached, from the price of a query, which is paid every time.
        """
        start = ReadStats(self.stats.requests, self.stats.bytes_read)
        delta = ReadStats()
        try:
            yield delta
        finally:
            delta.requests = self.stats.requests - start.requests
            delta.bytes_read = self.stats.bytes_read - start.bytes_read

    def _record(self, key: str, start: int | None, length: int, data: bytes) -> bytes:
        with self._lock:
            self.stats.requests += 1
            self.stats.bytes_read += len(data)
            if self.trace:
                self.reads.append(Read(key, start, length, len(data)))
        return data

    def get_range(self, key: str, start: int, length: int) -> bytes:
        return self._record(key, start, length, self.inner.get_range(key, start, length))

    def get_suffix(self, key: str, length: int) -> bytes:
        return self._record(key, None, length, self.inner.get_suffix(key, length))

    def put(self, key: str, data: bytes) -> None:
        self.inner.put(key, data)

    def delete(self, key: str) -> None:
        self.inner.delete(key)

    def list_keys(self, key: str = "") -> Iterator[str]:
        return self.inner.list_keys(key)

    def modified_at(self, key: str) -> float:
        return self.inner.modified_at(key)

    def size(self, key: str) -> int:
        return self.inner.size(key)

    def exists(self, key: str) -> bool:
        return self.inner.exists(key)
