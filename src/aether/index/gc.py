"""Deleting objects no manifest points at.

Segments are immutable but not eternal, and until now nothing ever removed
one. Three things leave objects behind that no manifest references:

- A crash between writing a segment and writing the manifest. The segment is
  complete and invisible, and the replay writes it again under the same key,
  so it is harmless but permanent.
- A time-based flush that seals offsets 100-101, crashes, and is replayed as
  100-104 under a different key. `Manifest.publish` correctly unlists the
  narrower one so its documents are not counted twice, and the object stays.
- Compaction, which retires ten segments on every merge by design.

The last of those is why this exists now. Without it, compaction trades a
request-count problem for a storage-growth problem.

## Why this is the one place that lists

The manifest exists so that queries never list a bucket: listing is slow,
costs a request per page, and can return objects that are still uploading.
None of that applies here, and more importantly the manifest is useless for
this job. An orphan is precisely an object no manifest mentions, so the only
way to find one is to look at what is actually stored.

## Why nothing recent is ever deleted

Being unreferenced is not the same as being dead. Two live situations produce
an object that no manifest names yet:

- An indexer has written a segment and has not published its manifest. That
  is one line of the flush apart, but a collector running in that window
  would delete a segment about to go live.
- A coordinator read the manifest a second ago and is still reading segments
  a newer manifest has dropped. Compaction produces exactly this, on every
  merge.

Age is what separates those from genuine garbage, so nothing younger than
`grace_seconds` is touched regardless of how unreferenced it looks. The
default is an hour, which is far longer than either window needs and costs
only some disk.

Dry-run is the default. Deleting has to be asked for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from aether.index.manifest import read_manifest
from aether.storage.base import ObjectStore

# Long enough that no in-flight reader or half-published flush can still be
# relying on an object, and short enough that garbage does not accumulate for
# a day. Nothing measures this; it is deliberately generous.
DEFAULT_GRACE_SECONDS = 3600.0

MANIFEST_SUFFIX = "current.json"


@dataclass
class Orphan:
    key: str
    bytes: int
    age_seconds: float


@dataclass
class CollectionResult:
    """What the collector found, and what it did about it."""

    scanned: int = 0
    # Objects the collector refused to consider: every segment any manifest
    # names, plus the manifests themselves. Named for what it counts rather
    # than "referenced", since a manifest is protected without being
    # referenced by anything.
    protected: int = 0
    manifests: list[str] = field(default_factory=list)
    deleted: list[Orphan] = field(default_factory=list)
    spared: list[Orphan] = field(default_factory=list)
    dry_run: bool = True

    @property
    def bytes_deleted(self) -> int:
        return sum(orphan.bytes for orphan in self.deleted)

    @property
    def bytes_spared(self) -> int:
        return sum(orphan.bytes for orphan in self.spared)

    def __str__(self) -> str:
        verb = "would delete" if self.dry_run else "deleted"
        return (
            f"scanned {self.scanned}, {self.protected} protected, "
            f"{verb} {len(self.deleted)} ({self.bytes_deleted / 1_048_576:.1f} MB), "
            f"spared {len(self.spared)} as too recent"
        )


def find_manifests(store: ObjectStore, prefix: str = "") -> list[str]:
    """Every manifest under a prefix.

    There is one per Kafka partition when the streaming indexer wrote the
    index, and a single top-level one when batch ingest did. Both are found,
    because missing one would make every segment it references look like an
    orphan, and the collector would delete a live index.
    """
    found = [
        key
        for key in store.list_keys(prefix)
        if key.endswith(MANIFEST_SUFFIX) or key.endswith("manifest.json")
    ]
    return sorted(found)


def referenced_keys(store: ObjectStore, manifests: list[str]) -> set[str]:
    """Every segment key any manifest names."""
    keys: set[str] = set()
    for manifest_key in manifests:
        for segment in read_manifest(store, manifest_key).segments:
            keys.add(segment.key)
    return keys


def collect(
    store: ObjectStore,
    *,
    prefix: str = "",
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
    delete: bool = False,
    now: float | None = None,
) -> CollectionResult:
    """Find, and optionally remove, objects no manifest references.

    Deletes only what is both unreferenced and older than `grace_seconds`.
    Manifests themselves are never candidates, whatever their age.
    """
    if grace_seconds < 0:
        raise ValueError(f"grace_seconds must not be negative, got {grace_seconds}")

    moment = time.time() if now is None else now
    manifests = find_manifests(store, prefix)
    live = referenced_keys(store, manifests)
    protected = live | set(manifests)

    result = CollectionResult(manifests=manifests, dry_run=not delete)

    for key in store.list_keys(prefix):
        result.scanned += 1
        if key in protected:
            result.protected += 1
            continue

        try:
            age = moment - store.modified_at(key)
            orphan = Orphan(key, store.size(key), age)
        except FileNotFoundError:
            # Deleted by something else between listing and inspecting. Not a
            # problem: the goal was for it to be gone.
            continue

        if age < grace_seconds:
            # Possibly a segment whose manifest has not landed yet, or one an
            # in-flight query is still reading.
            result.spared.append(orphan)
            continue

        result.deleted.append(orphan)
        if delete:
            store.delete(key)

    return result
