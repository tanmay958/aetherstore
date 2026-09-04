"""Index a CSV into many segments and publish a manifest.

The batch ancestor of the streaming indexer. It does what that will do, minus
Kafka: accumulate documents in memory, flush a segment when enough have piled
up, throw the memory away, repeat, and finally write the manifest that makes
all of it searchable at once.

Memory stays flat regardless of input size, because only one batch is ever
held. That is the same reason segments exist at all: RAM runs out, and object
storage cannot be appended to, so the only move is to write new files.

Segment names are derived from the row range they cover, not from a counter or
a timestamp:

    segments/000000000000-000000004999.seg

Deterministic naming is what will make a crashed indexer safe to replay. Re-running
the same input rewrites byte-identical objects at the same keys instead of
littering the bucket with duplicates, and duplicated documents would corrupt
BM25 silently rather than loudly. Nothing depends on that yet, and doing it
now costs nothing.

    python -m aether.index.ingest events.csv data/idx --docs-per-segment 5000
    python -m aether.index.ingest events.csv r2://aether/idx
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from aether.data.rees46 import iter_events
from aether.env import load_dotenv
from aether.index.manifest import Manifest, SegmentMeta, write_manifest
from aether.index.memory import MemoryIndex
from aether.index.segment import SegmentReader, write_segment
from aether.storage import open_store
from aether.storage.base import ObjectStore

DEFAULT_DOCS_PER_SEGMENT = 5_000


def segment_key(first_doc: int, last_doc: int) -> str:
    """Fixed width so keys sort lexicographically in document order."""
    return f"segments/{first_doc:012d}-{last_doc:012d}.seg"


def ingest(
    events,
    store: ObjectStore,
    *,
    docs_per_segment: int = DEFAULT_DOCS_PER_SEGMENT,
    on_flush=None,
) -> Manifest:
    """Consume events, writing a segment every `docs_per_segment` documents."""
    if docs_per_segment < 1:
        raise ValueError("docs_per_segment must be at least 1")

    segments: list[SegmentMeta] = []
    index = MemoryIndex()
    first_doc = 0
    seen = 0

    def flush() -> None:
        nonlocal index, first_doc
        if index.num_docs == 0:
            return
        key = segment_key(first_doc, seen - 1)
        data = write_segment(index)
        store.put(key, data)

        footer = SegmentReader(store, key).footer
        meta = SegmentMeta(
            key,
            footer.num_docs,
            len(data),
            footer.min_ts,
            footer.max_ts,
            first_offset=first_doc,
            last_offset=seen - 1,
        )
        segments.append(meta)
        if on_flush:
            on_flush(meta)

        # The memory is gone the moment it is durable. This is the whole
        # reason a long stream does not need a large machine.
        index = MemoryIndex()
        first_doc = seen

    for event in events:
        index.add(event)
        seen += 1
        if index.num_docs >= docs_per_segment:
            flush()
    flush()

    # The commit. Nothing above was searchable until this line lands, and
    # everything becomes searchable at the same instant.
    manifest = Manifest().with_segments(segments)
    write_manifest(store, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aether.index.ingest",
        description="Index a REES46 CSV into segments and publish a manifest.",
    )
    parser.add_argument("input", type=Path, help="REES46 .csv or .csv.gz")
    parser.add_argument("output", help="a directory, s3://bucket/prefix, or r2://...")
    parser.add_argument(
        "--docs-per-segment",
        type=int,
        default=DEFAULT_DOCS_PER_SEGMENT,
        help=f"documents per segment (default: {DEFAULT_DOCS_PER_SEGMENT:,})",
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N events")
    args = parser.parse_args(argv)
    load_dotenv()

    if not args.input.exists():
        parser.error(f"{args.input} not found. See docs/DATA.md for how to get it.")

    store = open_store(args.output)
    print(f"indexing {args.input} -> {args.output}")

    began = time.perf_counter()
    manifest = ingest(
        iter_events(args.input, limit=args.limit),
        store,
        docs_per_segment=args.docs_per_segment,
        on_flush=lambda meta: print(
            f"  flushed {meta.key}  {meta.docs:,} docs  {meta.bytes:,} B"
        ),
    )
    elapsed = time.perf_counter() - began

    print()
    print(f"  segments       {len(manifest.segments):,}")
    print(f"  documents      {manifest.docs:,}")
    print(f"  index size     {manifest.bytes:,} B")
    if elapsed > 0 and manifest.docs:
        print(f"  throughput     {manifest.docs / elapsed:,.0f} docs/sec")
    print(f"  elapsed        {elapsed:,.1f} s")
    print(f"  manifest       generation {manifest.generation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
