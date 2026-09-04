"""Build a segment file from a REES46 CSV.

    python -m aether.index.build tests/fixtures/rees46_sample.csv out.seg
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from aether.data.rees46 import iter_events
from aether.index.memory import build_index
from aether.index.segment import write_segment_file


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
    print(f"wrote {args.output}")
    print(f"  documents      {index.num_docs:,}")
    print(f"  terms          {index.num_terms:,}")
    print(f"  postings       {index.num_postings:,}")
    print(f"  avg doc length {index.avg_doc_length:.1f} terms")
    print(f"  indexed in     {build_ms:,.1f} ms")
    print(f"  written in     {write_ms:,.1f} ms")
    print(f"  segment size   {written:,} B")
    # Worth watching from here on. JSON has no business winning this, and it
    # does not; step 5 is where the number starts moving in the right
    # direction.
    print(f"  source size    {raw:,} B  ({written / raw:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
