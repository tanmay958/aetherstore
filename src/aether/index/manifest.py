"""The manifest: which segments are currently live.

An index is not a file. It is a pile of segment files, and something has to
say which ones count. That something is a small JSON object, and it is the
most important small idea in the whole design:

    Writing a segment does not make its data searchable. Adding it to the
    manifest does.

That single property is what buys atomicity on storage that offers no
transactions. Uploading two megabytes takes seconds and can fail halfway, but
until the manifest names the result nothing can see it, so a half-written
segment is invisible garbage rather than a corrupt index. Then one small write
flips ten thousand documents into visibility at once.

It also replaces LIST. Asking the bucket what objects exist is slow, costs a
request per page, and worst of all reports objects that are mid-upload and not
meant to be live. Reading one small file instead is a single request and
returns a set that is consistent by construction.

The per-segment metadata copied in here exists so a query can discard whole
segments before touching them. A search limited to the last hour can drop
every segment whose newest event is three days old using nothing but this
file, for zero additional requests.

Later, when several indexers run at once, there will be one manifest per Kafka
partition rather than one overall, because consumer group assignment
guarantees a single owner per partition and therefore a single writer per
file. That is how single-writer safety is inherited rather than built.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable

from aether.storage.base import ObjectStore

MANIFEST_VERSION = 1
DEFAULT_MANIFEST_KEY = "manifest.json"


@dataclass(frozen=True)
class SegmentMeta:
    """What the manifest records about one segment.

    Everything here is copied from the segment's own footer at write time.
    Duplicating it is deliberate: the point is that a coordinator can plan and
    prune a query having read only the manifest.
    """

    key: str
    docs: int
    bytes: int
    min_ts: int
    max_ts: int
    # The source range this segment covers: Kafka offsets for the streaming
    # indexer, row numbers for batch ingest. Recorded so that republishing a
    # range can evict whatever previously covered it. Optional because a
    # segment produced some other way may have no natural range.
    first_offset: int | None = None
    last_offset: int | None = None

    def covers(self, first: int, last: int) -> bool:
        """Whether this segment's source range intersects [first, last].

        Two segments covering overlapping ranges hold some of the same
        documents, and an index listing both counts those documents twice.
        BM25 then scores against inflated frequencies and the relevance is
        quietly wrong, with nothing anywhere reporting an error.
        """
        if self.first_offset is None or self.last_offset is None:
            return False
        return self.first_offset <= last and first <= self.last_offset

    def overlaps(self, start: int | None, end: int | None) -> bool:
        """Whether this segment could hold anything in a time window.

        The cheapest optimization available: pure arithmetic on data already
        in memory, discarding whole segments for no requests at all.
        """
        if start is not None and self.max_ts < start:
            return False
        if end is not None and self.min_ts > end:
            return False
        return True


@dataclass(frozen=True)
class Manifest:
    """The live set."""

    segments: tuple[SegmentMeta, ...] = ()
    generation: int = 0
    version: int = MANIFEST_VERSION

    @property
    def docs(self) -> int:
        return sum(segment.docs for segment in self.segments)

    @property
    def bytes(self) -> int:
        return sum(segment.bytes for segment in self.segments)

    def publish(self, segment: SegmentMeta) -> Manifest:
        """Add a segment, evicting anything it supersedes.

        A replay can produce a segment covering a wider range than the one it
        is redoing: an interrupted flush may have sealed offsets 100 to 101,
        while the replay reaches 100 to 104 before flushing. Appending both
        would list offsets 100 and 101 twice.

        Evicting by range rather than by key is what makes that safe. The
        superseded object stays in storage, unreferenced and invisible, until
        a collector removes it, which is the cost of a non-deterministic flush
        boundary and the reason count-based flushing is the default.
        """
        kept = [
            existing
            for existing in self.segments
            if existing.key != segment.key
            and not (
                segment.first_offset is not None
                and existing.covers(segment.first_offset, segment.last_offset)
            )
        ]
        return self.with_segments([*kept, segment])

    def with_segments(self, segments: Iterable[SegmentMeta]) -> Manifest:
        """A new manifest naming a different set, one generation later.

        Generations are not used for coordination yet. They exist because a
        rebalancing Kafka consumer group can briefly leave two writers
        believing they own the same partition, and a monotonic number is what
        lets a zombie be detected rather than silently overwriting a newer
        state.
        """
        return Manifest(tuple(segments), self.generation + 1, self.version)

    def to_json(self) -> bytes:
        return json.dumps(
            {
                "version": self.version,
                "generation": self.generation,
                "segments": [asdict(segment) for segment in self.segments],
            },
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> Manifest:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest is not valid JSON: {exc}") from exc

        version = payload.get("version")
        if version != MANIFEST_VERSION:
            raise ValueError(
                f"manifest version {version} is not supported "
                f"(this build reads version {MANIFEST_VERSION})"
            )
        return cls(
            tuple(SegmentMeta(**entry) for entry in payload.get("segments", [])),
            payload.get("generation", 0),
        )


PARTITION_MANIFEST_PREFIX = "manifests/"


def partition_manifest_keys(
    store: ObjectStore, prefix: str = PARTITION_MANIFEST_PREFIX
) -> list[str]:
    """Every per-partition manifest under a prefix.

    The streaming indexer writes one manifest per Kafka partition, because
    single-writer safety is inherited from the log: exactly one consumer owns
    a partition, so exactly one process writes that manifest and no locking is
    needed anywhere. A shared manifest would throw that away and need
    coordination to put it back.

    The read side therefore has to gather them, and it cannot know how many
    partitions there are without looking. This is the one query-time listing
    in the engine, and it is one request against a handful of keys.
    """
    return sorted(key for key in store.list_keys(prefix) if key.endswith(".json"))


def union_manifests(manifests: Iterable[Manifest]) -> Manifest:
    """One live set spanning several partitions.

    Concatenation, deduplicated by object key, and deliberately **not**
    `publish`. Publish evicts by offset range, which is right within a
    partition and catastrophic across them: offsets are per-partition, so
    partition 0 and partition 1 both hold a segment covering offsets 0 to
    1999, and each would evict the other. The result would silently serve a
    twelfth of the index.
    """
    seen: dict[str, SegmentMeta] = {}
    generation = 0
    for manifest in manifests:
        generation = max(generation, manifest.generation)
        for segment in manifest.segments:
            seen.setdefault(segment.key, segment)
    return Manifest(tuple(seen.values()), generation, MANIFEST_VERSION)


def read_manifest(store: ObjectStore, key: str = DEFAULT_MANIFEST_KEY) -> Manifest:
    """Load the live set. An absent manifest is an empty index, not an error."""
    try:
        raw = store.get_range(key, 0, store.size(key))
    except FileNotFoundError:
        return Manifest()
    return Manifest.from_json(raw)


def write_manifest(
    store: ObjectStore, manifest: Manifest, key: str = DEFAULT_MANIFEST_KEY
) -> None:
    """Publish a live set.

    This is the commit. Every segment named here becomes searchable at the
    instant this single small write lands, and anything not named stays
    invisible no matter how completely it was uploaded.
    """
    store.put(key, manifest.to_json())
