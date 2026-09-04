"""Search either a REES46 CSV or a prebuilt segment file.

Given a .seg it opens the segment; given anything else it indexes the CSV in
memory first. The query path afterwards is identical, because both satisfy
SearchableIndex, which is the point: a caller never learns where the postings
came from.

    python -m aether.index.search tests/fixtures/rees46_sample.csv "samsung smartphone"
    python -m aether.index.search out.seg "samsung smartphone"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from aether.data.rees46 import iter_events
from aether.index.analyzer import tokenize
from aether.index.base import SearchableIndex
from aether.index.memory import build_index
from aether.index.segment import SegmentReader


def load(path: Path) -> tuple[SearchableIndex, str]:
    """Open a segment, or index a CSV in memory."""
    if path.suffix == ".seg":
        return SegmentReader.open(path), "segment"
    return build_index(iter_events(path)), "memory index"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aether.index.search",
        description="Build an in-memory inverted index and query it.",
    )
    parser.add_argument("input", type=Path, help="REES46 .csv/.csv.gz or a .seg")
    parser.add_argument("query", help="search terms")
    parser.add_argument(
        "--or", dest="use_or", action="store_true", help="match any term instead of all"
    )
    parser.add_argument("--top", type=int, default=10, help="results to display")
    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(f"{args.input} not found. See docs/DATA.md for how to get it.")

    started = time.perf_counter()
    index, kind = load(args.input)
    build_ms = (time.perf_counter() - started) * 1000

    print(
        f"{kind}: {index.num_docs:,} docs, {index.num_terms:,} terms, "
        f"{index.num_postings:,} postings   opened in {build_ms:,.1f} ms"
    )
    print(f"       {index.avg_doc_length:.1f} terms per document on average")
    print()

    terms = tokenize(args.query)
    if not terms:
        print(f'query "{args.query}" contains no indexable terms')
        return 1

    print(f'query "{args.query}"  ->  {terms}')
    for term in terms:
        df = index.df(term)
        # Rare terms are the informative ones, which is what BM25 will
        # formalize in step 6.
        note = "  (not in index)" if df == 0 else ""
        print(f"    {term:<20} df {df:>6,}{note}")

    started = time.perf_counter()
    hits = index.search_or(args.query) if args.use_or else index.search_and(args.query)
    query_ms = (time.perf_counter() - started) * 1000

    mode = "OR " if args.use_or else "AND"
    print(f"    {mode:<20} {len(hits):>6,} docs in {query_ms:.3f} ms")
    print()

    if not hits:
        print("  no matches")
        return 0

    # Ordered by document id, not relevance: ranking arrives with BM25.
    for doc_id in hits[: args.top]:
        doc = index.document(doc_id)
        title = doc["title"] or "(no title)"
        price = f"${doc['price']:,.2f}" if doc["price"] is not None else ""
        print(
            f"  doc {doc_id:>6}  {title:<42} {doc['event_type']:<16} "
            f"{doc['category'] or '':<34} {price:>10}"
        )
    if len(hits) > args.top:
        print(f"  ... and {len(hits) - args.top:,} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
