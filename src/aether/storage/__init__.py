"""Object storage: the byte-range interface segments are read through."""

from aether.storage.base import ObjectStore
from aether.storage.counting import CountingStore, ReadStats
from aether.storage.local import LocalStore

__all__ = ["ObjectStore", "LocalStore", "CountingStore", "ReadStats"]
