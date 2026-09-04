"""Build a segment file from a REES46 CSV.

    python -m aether.index.build events.csv out.seg
    python -m aether.index.build events.csv s3://aether/segments/0.seg
    python -m aether.index.build events.csv r2://aether/segments/0.seg
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from aether.data.rees46 import iter_events
from aether.index.memory import build_index
from aether.index.segment import FOOTER_SIZE, SegmentReader, write_segment
from aether.storage import open_object
from aether.env import load_dotenv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aether.index.build",
        description="Index a REES46 CSV into a segment file.",
    )
    parser.add_argument("input", type=Path, help="REES46 .csv or .csv.gz")
    parser.add_argument(
        "output", help="where to write: a path, s3://bucket/key, or r2://bucket/key"
    )
    parser.add_argument("--limit", type=int, default=None, help="index at most N events")
    args = parser.parse_args(argv)
    load_dotenv()

    if not args.input.exists():
        parser.error(f"{args.input} not found. See docs/DATA.md for how to get it.")

    started = time.perf_counter()
    index = build_index(iter_events(args.input, limit=args.limit))
    build_ms = (time.perf_counter() - started) * 1000

    # The same call whether this lands on disk, in MinIO, in R2, or in S3.
    # That is the whole return on defining ObjectStore back in step 3.
    store, key = open_object(args.output)
    data = write_segment(index)

    started = time.perf_counter()
    store.put(key, data)
    write_ms = (time.perf_counter() - started) * 1000
    written = len(data)

    raw = args.input.stat().st_size
    footer = SegmentReader(store, key).footer

    print(f"wrote {args.output}")
    print(f"  documents      {index.num_docs:,}")
    print(f"  terms          {index.num_terms:,}")
    print(f"  postings       {index.num_postings:,}")
    print(f"  avg doc length {index.avg_doc_length:.1f} terms")
    print(f"  indexed in     {build_ms:,.1f} ms")
    print(f"  uploaded in    {write_ms:,.1f} ms")
    print()
    print(f"  segment size   {written:,} B")
    for name, size in (
        ("postings", footer.postings_length),
        ("docstore", footer.docstore_length),
        ("termdict", footer.termdict_length),
        ("hotcache", footer.hotcache_length),
        ("footer", FOOTER_SIZE),
    ):
        # The hotcache is the number to watch: it is read once per segment and
        # then cached forever, so it is the memory price of never having to
        # fetch the rest.
        print(f"    {name:<12} {size:>10,} B  {100 * size / written:>5.1f}%")
    print()
    # The codec metric. A posting is one document id plus one frequency, so
    # fixed-width 32-bit storage costs 8 bytes; anything below that is what
    # delta encoding and varints bought.
    per_posting = footer.postings_length / index.num_postings if index.num_postings else 0
    print(f"  bytes/posting  {per_posting:.2f}  (8.00 at fixed width)")
    print(f"  source size    {raw:,} B  ({written / raw:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
