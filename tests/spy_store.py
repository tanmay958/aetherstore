"""A store that counts what the reader asks for.

Step 3 claims the reader never loads a whole segment. That claim is only worth
anything if something checks it, so tests wrap a real store in this and assert
on request counts and bytes moved.

Step 4 promotes this idea into `src` as real instrumentation, because the
request count is the number that matters on object storage: it is what you pay
in both latency and money, and it is the number that carries over unchanged
from a local disk to S3.
"""

from aether.storage.base import ObjectStore


class CountingStore(ObjectStore):
    def __init__(self, inner: ObjectStore) -> None:
        self.inner = inner
        self.reads = 0
        self.bytes_read = 0

    def reset(self) -> None:
        self.reads = 0
        self.bytes_read = 0

    def _record(self, data: bytes) -> bytes:
        self.reads += 1
        self.bytes_read += len(data)
        return data

    def get_range(self, key: str, start: int, length: int) -> bytes:
        return self._record(self.inner.get_range(key, start, length))

    def get_suffix(self, key: str, length: int) -> bytes:
        return self._record(self.inner.get_suffix(key, length))

    def put(self, key: str, data: bytes) -> None:
        self.inner.put(key, data)

    def size(self, key: str) -> int:
        return self.inner.size(key)

    def exists(self, key: str) -> bool:
        return self.inner.exists(key)
