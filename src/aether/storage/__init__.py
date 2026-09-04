"""Object storage: the byte-range interface segments are read through."""

from aether.storage.base import ObjectStore
from aether.storage.local import LocalStore

__all__ = ["ObjectStore", "LocalStore"]
