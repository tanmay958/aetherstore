"""Merging small segments into larger ones.

The indexer seals a segment every so many documents and never revisits it, so
segments accumulate without limit: a million documents is a hundred of them,
and a month of REES46 would be several thousand. Every query opens every live
segment before it can search, at two requests each, and that cost is paid
again by every cold process.

    opening 100 segments   201 requests   ~2,500 ms over R2
    merged to 10 segments   21 requests   ~  250 ms

Measured against a laptop's ~200 ms per range read, that is most of a cold
query. It is the largest remaining win in the engine, and it matters most in a
serverless deployment where every process is cold.

## Merging by re-indexing

Sources are read back as documents and fed through `MemoryIndex`, the same
path ordinary indexing takes, then written with `write_segment`.

The faster alternative is to merge posting lists directly, concatenating them
with document ids remapped for their new positions. That is what a mature
engine does, and it would take roughly 0.3 seconds where this takes 1.7 for a
hundred thousand documents. It is not worth it here: compaction is a
background job with no latency budget, and remapping ids across segments is
the kind of code whose bugs are silent and show up later as wrong relevance.
Re-indexing cannot produce a segment structurally different from a freshly
built one, and the oracle suite already covers that path.

It also reads *less* than a posting-list merge would. Documents have to be
read either way to write the merged docstore; re-indexing simply never touches
the source postings sections at all.

## Only adjacent ranges

`SegmentMeta` records the offset range a segment covers, and `Manifest.publish`
evicts by range overlap so a replayed batch supersedes the partial one it
redoes. Merging segments that are not adjacent would produce a range with a
hole in it, and a later replay landing inside that hole would evict a merged
segment holding thousands of unrelated documents. Sources are therefore sorted
by offset and only consecutive runs are merged.

## What a crash leaves behind

    1. read the source documents
    2. PUT the merged segment      <- crash: unreferenced object, GC removes it
    3. PUT the manifest            <- the commit; every query now sees the merge
    4. leave the sources alone     <- deliberately

Step 4 is the one that is easy to get wrong. A coordinator that read the
manifest a moment ago is still reading those sources, and deleting them
underneath it fails a live query for no reason. Retiring them is the
collector's job, after a grace period long enough for in-flight readers to
finish. See `gc.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from aether.index.manifest import (
    DEFAULT_MANIFEST_KEY,
    Manifest,
    SegmentMeta,
    read_manifest,
    write_manifest,
)
from aether.index.memory import MemoryIndex
from aether.index.segment import SegmentReader, write_segment
from aether.storage.base import ObjectStore

# How many segments make a merge worth doing. Below this the work costs more
# than the requests it saves.
DEFAULT_MERGE_FACTOR = 10

# A merged segment stops growing here. Without a ceiling, repeated compaction
# converges on one enormous segment, and every subsequent merge would rewrite
# the entire index to absorb a few new documents.
DEFAULT_MAX_MERGED_BYTES = 256 * 1024 * 1024

# Segments within this factor of each other in size count as one tier. Merging
# a 100 MB segment with a 1 MB one rewrites 101 MB to save a single request.
SIZE_TIER_RATIO = 4.0


@dataclass
class CompactionPlan:
    """A set of merges to perform, decided before anything is written."""

    groups: list[list[SegmentMeta]] = field(default_factory=list)

    @property
    def segments_in(self) -> int:
        return sum(len(group) for group in self.groups)

    @property
    def segments_out(self) -> int:
        return len(self.groups)

    @property
    def bytes_in(self) -> int:
        return sum(segment.bytes for group in self.groups for segment in group)

    def __bool__(self) -> bool:
        return bool(self.groups)


@dataclass
class CompactionResult:
    merged: list[SegmentMeta] = field(default_factory=list)
    retired: list[SegmentMeta] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def documents(self) -> int:
        return sum(segment.docs for segment in self.merged)

    def __str__(self) -> str:
        return (
            f"{len(self.retired)} segments -> {len(self.merged)}, "
            f"{self.documents:,} documents, {self.elapsed_seconds:.1f}s"
        )


def plan_compaction(
    manifest: Manifest,
    *,
    merge_factor: int = DEFAULT_MERGE_FACTOR,
    max_merged_bytes: int = DEFAULT_MAX_MERGED_BYTES,
) -> CompactionPlan:
    """Decide what to merge, without touching storage.

    Size-tiered and adjacency-constrained. Segments are walked in offset
    order, and a run accumulates while its members stay within
    `SIZE_TIER_RATIO` of each other and the total stays under
    `max_merged_bytes`. A run of at least `merge_factor` segments becomes a
    merge; anything shorter is left alone.

    Separating the decision from the work keeps the policy testable without a
    store, and lets the CLI show what it would do before doing it.
    """
    if merge_factor < 2:
        raise ValueError(f"merge_factor must be at least 2, got {merge_factor}")

    ordered = sorted(
        manifest.segments,
        key=lambda segment: (
            segment.first_offset if segment.first_offset is not None else 0,
            segment.key,
        ),
    )

    plan = CompactionPlan()
    run: list[SegmentMeta] = []
    run_bytes = 0

    def close_run() -> None:
        nonlocal run, run_bytes
        if len(run) >= merge_factor:
            plan.groups.append(run)
        run = []
        run_bytes = 0

    for segment in ordered:
        if run:
            smallest = min(s.bytes for s in run)
            largest = max(s.bytes for s in run)
            same_tier = (
                segment.bytes <= largest * SIZE_TIER_RATIO
                and segment.bytes * SIZE_TIER_RATIO >= smallest
            )
            if not same_tier or run_bytes + segment.bytes > max_merged_bytes:
                close_run()
        run.append(segment)
        run_bytes += segment.bytes
        # A run is capped at merge_factor rather than growing to whatever
        # happens to be the same size. Without this a thousand uniform
        # segments become a single merge that rewrites the entire index, which
        # is the unbounded behaviour a tiered policy exists to avoid: work per
        # run stays fixed, and repeated passes converge instead of one pass
        # doing everything.
        if len(run) == merge_factor:
            close_run()
    close_run()

    return plan


def merged_key(group: list[SegmentMeta]) -> str:
    """Name a merged segment after the range it now covers.

    Same discipline as everywhere else: derived from the data rather than from
    a counter or a clock, so re-running an identical merge overwrites an
    identical object instead of leaving a second copy.
    """
    first = group[0].key.rsplit("/", 1)
    directory = f"{first[0]}/" if len(first) == 2 else ""
    lo = group[0].first_offset
    hi = group[-1].last_offset
    if lo is None or hi is None:
        # No offsets recorded, so fall back to the source key range, which is
        # still derived from the inputs rather than invented.
        return f"{directory}merged-{group[0].key.rsplit('/', 1)[-1]}"
    return f"{directory}{lo:020d}-{hi:020d}.seg"


def merge_group(
    store: ObjectStore, group: list[SegmentMeta], manifest_key: str
) -> SegmentMeta:
    """Merge one run of segments and publish the result.

    Returns the merged segment's metadata. The sources are left in storage
    untouched; only the manifest stops pointing at them.
    """
    index = MemoryIndex()
    for segment in group:
        reader = SegmentReader(store, segment.key)
        for doc_id in range(reader.num_docs):
            index.add(reader.document(doc_id))

    key = merged_key(group)
    data = write_segment(index)
    store.put(key, data)

    footer = SegmentReader(store, key).footer
    merged = SegmentMeta(
        key,
        footer.num_docs,
        len(data),
        footer.min_ts,
        footer.max_ts,
        first_offset=group[0].first_offset,
        last_offset=group[-1].last_offset,
    )

    # The commit. Read the manifest fresh rather than trusting a copy taken
    # before the merge started, so a segment published while this ran is not
    # dropped.
    live = read_manifest(store, manifest_key)
    retired = {segment.key for segment in group}
    kept = [s for s in live.segments if s.key not in retired and s.key != key]
    write_manifest(store, live.with_segments([*kept, merged]), manifest_key)
    return merged


def compact(
    store: ObjectStore,
    *,
    manifest_key: str = DEFAULT_MANIFEST_KEY,
    merge_factor: int = DEFAULT_MERGE_FACTOR,
    max_merged_bytes: int = DEFAULT_MAX_MERGED_BYTES,
    on_merge=None,
) -> CompactionResult:
    """Plan and perform compaction against one manifest."""
    began = time.perf_counter()
    manifest = read_manifest(store, manifest_key)
    plan = plan_compaction(
        manifest, merge_factor=merge_factor, max_merged_bytes=max_merged_bytes
    )

    result = CompactionResult()
    for group in plan.groups:
        merged = merge_group(store, group, manifest_key)
        result.merged.append(merged)
        result.retired.extend(group)
        if on_merge:
            on_merge(group, merged)

    result.elapsed_seconds = time.perf_counter() - began
    return result
