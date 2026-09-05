"""An object store backed by the local filesystem.

Stands in for S3 or R2 during development and in tests. Range reads become
seeks, which are roughly three hundred times faster than a network round trip,
so this store will make the query engine look far better than it is. Step 4
counts requests rather than only timing them for exactly that reason: the
request count is the number that transfers to real object storage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from aether.storage.base import ObjectStore


class LocalStore(ObjectStore):
    """Objects are files under a root directory, keyed by relative path."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        root = self.root.resolve()
        # Keys come from manifests, which will eventually be written by other
        # processes. A key of "../../etc/passwd" should not resolve outside
        # the store.
        if not path.is_relative_to(root):
            raise ValueError(f"key {key!r} escapes the store root")
        return path

    def get_range(self, key: str, start: int, length: int) -> bytes:
        with self._path(key).open("rb") as handle:
            handle.seek(start)
            return handle.read(length)

    def get_suffix(self, key: str, length: int) -> bytes:
        path = self._path(key)
        with path.open("rb") as handle:
            # Negative offsets from the end mirror "Range: bytes=-N", which
            # needs no prior knowledge of the object's size.
            handle.seek(-min(length, path.stat().st_size), 2)
            return handle.read()

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        root = self.root.resolve()
        if not root.is_dir():
            return
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            # Keys are posix-style relative paths, so a listing produced here
            # is interchangeable with one produced by S3.
            key = path.relative_to(root).as_posix()
            if key.startswith(prefix):
                yield key

    def modified_at(self, key: str) -> float:
        return self._path(key).stat().st_mtime

    def size(self, key: str) -> int:
        return self._path(key).stat().st_size

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()
