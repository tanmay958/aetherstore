"""Search a REES46 CSV or a prebuilt segment, and report what it cost.

Given a .seg it opens the segment through a counting store; given anything
else it indexes the CSV in memory first. The query path afterwards is
identical, because both satisfy SearchableIndex, which is the point: a caller
never learns where the postings came from.

For segments the cost is reported in requests, not only milliseconds. Locally
every range read is a seek and therefore nearly free, so timings here flatter
the engine enormously. The request count is the number that carries over to
object storage unchanged, where each one is a 20-50ms round trip that costs
money regardless of how many bytes it returns.

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
from aether.storage import CountingStore, LocalStore, ReadStats


def load(path: Path) -> tuple[SearchableIndex, str, CountingStore | None]:
    """Open a segment through a counting store, or index a CSV in memory."""
    if path.suffix == ".seg":
        store = CountingStore(LocalStore(path.parent))
        return SegmentReader(store, path.name), "segment", store
    return build_index(iter_events(path)), "memory index", None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aether.index.search",
        description="Build or open an index and query it.",
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
    index, kind, store = load(args.input)
    open_ms = (time.perf_counter() - started) * 1000
    # A snapshot, not the live object: store.stats keeps accumulating.
    open_cost = ReadStats(store.stats.requests, store.stats.bytes_read) if store else None

    print(
        f"{kind}: {index.num_docs:,} docs, {index.num_terms:,} terms, "
        f"{index.num_postings:,} postings   opened in {open_ms:,.1f} ms"
    )
    print(f"       {index.avg_doc_length:.1f} terms per document on average")
    print()

    terms = tokenize(args.query)
    if not terms:
        print(f'query "{args.query}" contains no indexable terms')
        return 1

    print(f'query "{args.query}"  ->  {terms}')

    # Costs are gathered per stage so the report separates what is paid once
    # from what is paid on every query.
    noop = CountingStore(LocalStore(".")) if store is None else store
    with noop.measure() as lookup_cost:
        term_dfs = [(term, index.df(term)) for term in terms]

    for term, df in term_dfs:
        note = "  (not in index)" if df == 0 else ""
        print(f"    {term:<20} df {df:>6,}{note}")

    mode = "or" if args.use_or else "and"
    with noop.measure() as match_cost:
        started = time.perf_counter()
        result = index.search(args.query, top_k=args.top, mode=mode)
        query_ms = (time.perf_counter() - started) * 1000

    label = "OR " if args.use_or else "AND"
    print(f"    {label:<20} {result.total:>6,} docs in {query_ms:.3f} ms")
    print()

    if not result.hits:
        print("  no matches")
        return 0

    with noop.measure() as fetch_cost:
        shown = [(hit, index.document(hit.doc_id)) for hit in result.hits]

    print(f"  {'score':>7}  {'doc':>6}  title")
    for hit, doc in shown:
        title = doc["title"] or "(no title)"
        price = f"${doc['price']:,.2f}" if doc["price"] is not None else ""
        print(
            f"  {hit.score:>7.3f}  {hit.doc_id:>6}  {title:<42} "
            f"{doc['event_type']:<16} {doc['category'] or '':<28} {price:>10}"
        )
    if result.total > len(result.hits):
        print(f"  ... and {result.total - len(result.hits):,} more")

    if store is not None:
        print()
        print("  storage cost")
        print(f"    open (once)      {open_cost}")
        print(f"    term lookup      {lookup_cost}")
        print(f"    posting lists    {match_cost}")
        print(f"    fetch {len(shown):>2} docs     {fetch_cost}")
        print(f"    segment on disk  {store.size(args.input.name):,} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
