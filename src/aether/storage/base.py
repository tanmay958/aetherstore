"""The storage interface segments are read through.

Five methods, and the important one is `get_range`. Everything the query path
does is expressed as "give me bytes X through Y of this object", because that
is the only primitive object storage actually offers and it is the primitive
the whole segment format is designed around.

Keeping this abstract while everything is still local is the point. `LocalStore`
uses `seek`, and an S3-backed store issues an HTTP Range header, and neither
the segment reader nor its tests can tell the difference. That is what turns
"run this on real object storage" from a rewrite into a config change.

The cost model differs wildly between implementations and the interface
deliberately hides it. On a local disk a range read is ~100 microseconds and
free. On S3 it is 20 to 50 milliseconds and costs money per request,
regardless of whether it returns 64 bytes or 8 megabytes. Code written against
this interface should therefore minimize the *number* of calls, not the bytes
returned, and should happily fetch data it will discard in order to avoid one
extra round trip.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator


class ObjectStore(ABC):
    """Read and write immutable objects, addressed by key."""

    @abstractmethod
    def get_range(self, key: str, start: int, length: int) -> bytes:
        """Bytes [start, start + length) of an object.

        Reading past the end returns what exists rather than raising, matching
        how HTTP range requests behave.
        """

    @abstractmethod
    def get_suffix(self, key: str, length: int) -> bytes:
        """The last `length` bytes of an object.

        Separate from `get_range` because it can be issued without knowing the
        object's size, which is exactly what `Range: bytes=-64` does over HTTP.
        That is what lets a reader locate a segment's footer, and therefore
        every other section, in a single request against a file it has never
        seen and whose length it does not know.
        """

    @abstractmethod
    def put(self, key: str, data: bytes) -> None:
        """Write an object whole.

        There is no append and no partial update, because object storage
        offers neither. That constraint is why segments are immutable.
        """

    @abstractmethod
    def list_keys(self, prefix: str = "") -> Iterator[str]:
        """Every key under a prefix.

        Listing is avoided everywhere else in this engine, and the manifest
        exists precisely so that queries never need it: it is slow, it costs a
        request per page, and it can return objects that are still being
        uploaded.

        Garbage collection is the one job that genuinely requires it. An
        object no manifest mentions is invisible by definition, so the
        manifest cannot be used to find it, and something has to look at what
        is actually there.
        """

    @abstractmethod
    def modified_at(self, key: str) -> float:
        """When an object was last written, in epoch seconds.

        Used only by the collector, to leave recent objects alone. An object
        can be unreferenced because it is genuinely dead, or because an
        indexer wrote it moments ago and has not published its manifest yet,
        and age is what tells those apart.
        """

    @abstractmethod
    def size(self, key: str) -> int:
        """Length of an object in bytes."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove an object. Deleting something absent is not an error.

        Segments are immutable, but they are not eternal: compaction merges
        several into one and then removes the originals, and a garbage
        collector eventually removes objects no manifest references. Both need
        this, and both must tolerate the object already being gone, because a
        retried delete is the normal case rather than the exception.
        """

    @abstractmethod
    def exists(self, key: str) -> bool: ...
