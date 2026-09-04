"""Build a segment file from a REES46 CSV.

    python -m aether.index.build tests/fixtures/rees46_sample.csv out.seg
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from aether.data.rees46 import iter_events
from aether.index.memory import build_index
from aether.index.segment import FOOTER_SIZE, SegmentReader, write_segment_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aether.index.build",
        description="Index a REES46 CSV into a segment file.",
    )
    parser.add_argument("input", type=Path, help="REES46 .csv or .csv.gz")
    parser.add_argument("output", type=Path, help="segment file to write")
    parser.add_argument("--limit", type=int, default=None, help="index at most N events")
    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(f"{args.input} not found. See docs/DATA.md for how to get it.")

    started = time.perf_counter()
    index = build_index(iter_events(args.input, limit=args.limit))
    build_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    written = write_segment_file(index, args.output)
    write_ms = (time.perf_counter() - started) * 1000

    raw = args.input.stat().st_size
    footer = SegmentReader.open(args.output).footer

    print(f"wrote {args.output}")
    print(f"  documents      {index.num_docs:,}")
    print(f"  terms          {index.num_terms:,}")
    print(f"  postings       {index.num_postings:,}")
    print(f"  avg doc length {index.avg_doc_length:.1f} terms")
    print(f"  indexed in     {build_ms:,.1f} ms")
    print(f"  written in     {write_ms:,.1f} ms")
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
    # Step 5 is where this number starts moving in the right direction:
    # postings are still plain 32-bit ints and the docstore is raw JSON.
    print(f"  source size    {raw:,} B  ({written / raw:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
