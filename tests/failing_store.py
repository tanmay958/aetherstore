"""A store that fails on a chosen write, to simulate a crash mid-flush.

The indexer's correctness argument is entirely about what happens when the
process dies between two writes. Asserting that requires being able to kill it
at an exact point, which a real crash cannot give you reproducibly.
"""

from aether.storage.base import ObjectStore


class CrashAfter(ObjectStore):
    """Passes the first `writes` puts through, then raises on the next one."""

    class Crash(RuntimeError):
        pass

    def __init__(self, inner: ObjectStore, writes: int) -> None:
        self.inner = inner
        self.remaining = writes
        self.written: list[str] = []

    def put(self, key: str, data: bytes) -> None:
        if self.remaining <= 0:
            raise self.Crash(f"crashed before writing {key}")
        self.remaining -= 1
        self.written.append(key)
        self.inner.put(key, data)

    def get_range(self, key: str, start: int, length: int) -> bytes:
        return self.inner.get_range(key, start, length)

    def get_suffix(self, key: str, length: int) -> bytes:
        return self.inner.get_suffix(key, length)

    def delete(self, key: str) -> None:
        self.inner.delete(key)

    def size(self, key: str) -> int:
        return self.inner.size(key)

    def exists(self, key: str) -> bool:
        return self.inner.exists(key)
