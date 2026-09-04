"""Query an index, a single segment, or a CSV, and report what it cost.

Three kinds of target, one query path:

    events.csv                       index it in memory first
    out.seg                          one segment, read by byte range
    data/idx  |  r2://aether/idx     a whole index: manifest, many segments

Costs are reported in requests, not only milliseconds. Against local files
every read is a seek and the timings flatter the engine enormously; measured
from a laptop, R2 answers a range read in roughly 200 ms regardless of whether
it returns 64 bytes or 4 kilobytes. The request count is the number that
carries over unchanged, and it is what you pay in both latency and money.

    python -m aether.index.search data/idx "samsung smartphone"
    python -m aether.index.search r2://aether/idx "samsung" --or --global-stats
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from aether.data.rees46 import iter_events
from aether.env import load_dotenv
from aether.index.analyzer import tokenize
from aether.index.coordinator import Coordinator
from aether.index.memory import build_index
from aether.index.segment import SegmentReader
from aether.storage import CountingStore, ReadStats, open_object, open_store


def _describe(doc: dict) -> str:
    title = doc["title"] or "(no title)"
    price = f"${doc['price']:,.2f}" if doc["price"] is not None else ""
    return (
        f"{title:<42} {doc['event_type']:<16} "
        f"{doc['category'] or '':<28} {price:>10}"
    )


def _snapshot(store: CountingStore | None) -> ReadStats:
    return ReadStats(store.stats.requests, store.stats.bytes_read) if store else ReadStats()


def run_index(target: str, args) -> int:
    """A manifest and many segments."""
    store = CountingStore(open_store(target))
    coordinator = Coordinator(store, max_workers=args.workers)

    manifest = coordinator.manifest
    if not manifest.segments:
        print(f"no manifest at {target}. Build one with: python -m aether.index.ingest")
        return 1

    open_cost = _snapshot(store)
    print(
        f"index: {len(manifest.segments):,} segments, {manifest.docs:,} docs, "
        f"{manifest.bytes:,} B   generation {manifest.generation}"
    )
    print(f"       manifest read: {open_cost}")
    print()
    print(f'query "{args.query}"  ->  {tokenize(args.query)}')

    result = coordinator.search(
        args.query,
        top_k=args.top,
        mode="or" if args.use_or else "and",
        start=args.since,
        end=args.until,
        global_stats=args.global_stats,
    )
    search_cost = _snapshot(store)

    print(f"    {result.stats}")
    print(f"    {result.total:,} matching documents")
    print()
    if not result.hits:
        print("  no matches")
        return 0

    docs = coordinator.documents(result.hits)
    fetch_cost = ReadStats(
        store.stats.requests - search_cost.requests,
        store.stats.bytes_read - search_cost.bytes_read,
    )

    print(f"  {'score':>7}  {'doc':>6}  {'segment':<22} title")
    for hit, doc in zip(result.hits, docs):
        segment = hit.segment.rsplit("/", 1)[-1][:22]
        print(f"  {hit.score:>7.3f}  {hit.doc_id:>6}  {segment:<22} {_describe(doc)}")
    if result.total > len(result.hits):
        print(f"  ... and {result.total - len(result.hits):,} more")

    print()
    print("  storage cost")
    print(f"    manifest         {open_cost}")
    print(f"    search           {ReadStats(search_cost.requests - open_cost.requests, search_cost.bytes_read - open_cost.bytes_read)}")
    print(f"    fetch {len(docs):>2} docs     {fetch_cost}")
    print(f"    total            {store.stats}")
    return 0


def run_segment(target: str, args) -> int:
    """One segment."""
    inner, key = open_object(target)
    store = CountingStore(inner)
    segment = SegmentReader(store, key)
    open_cost = _snapshot(store)

    print(
        f"segment: {segment.num_docs:,} docs, {segment.num_terms:,} terms, "
        f"{segment.num_postings:,} postings"
    )
    print(f"       open cost: {open_cost}")
    print()
    print(f'query "{args.query}"  ->  {tokenize(args.query)}')

    began = time.perf_counter()
    result = segment.search(args.query, top_k=args.top, mode="or" if args.use_or else "and")
    print(f"    {result.total:,} docs in {(time.perf_counter() - began) * 1000:.1f} ms")
    print()
    if not result.hits:
        print("  no matches")
        return 0

    print(f"  {'score':>7}  {'doc':>6}  title")
    for hit in result.hits:
        print(f"  {hit.score:>7.3f}  {hit.doc_id:>6}  {_describe(segment.document(hit.doc_id))}")
    print()
    print(f"  storage cost     {store.stats}  (segment is {store.size(key):,} B)")
    return 0


def run_csv(target: str, args) -> int:
    """A CSV, indexed in memory. No storage cost to report."""
    began = time.perf_counter()
    index = build_index(iter_events(Path(target)))
    print(
        f"memory index: {index.num_docs:,} docs, {index.num_terms:,} terms   "
        f"built in {(time.perf_counter() - began) * 1000:,.1f} ms"
    )
    print()
    print(f'query "{args.query}"  ->  {tokenize(args.query)}')

    result = index.search(args.query, top_k=args.top, mode="or" if args.use_or else "and")
    print(f"    {result.total:,} matching documents")
    print()
    for hit in result.hits:
        print(f"  {hit.score:>7.3f}  {hit.doc_id:>6}  {_describe(index.document(hit.doc_id))}")
    return 0 if result.hits else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aether.index.search",
        description="Query an index, a segment, or a CSV.",
    )
    parser.add_argument("target", help="a CSV, a .seg, or an index prefix")
    parser.add_argument("query", help="search terms")
    parser.add_argument("--or", dest="use_or", action="store_true", help="match any term")
    parser.add_argument("--top", type=int, default=10, help="results to display")
    parser.add_argument("--since", type=int, default=None, help="epoch lower bound")
    parser.add_argument("--until", type=int, default=None, help="epoch upper bound")
    parser.add_argument(
        "--global-stats",
        action="store_true",
        help="score against corpus-wide document frequencies: correct across "
        "segments, at the price of a second wave of requests",
    )
    parser.add_argument("--workers", type=int, default=16, help="fan-out concurrency")
    args = parser.parse_args(argv)
    load_dotenv()

    target = args.target
    local = "://" not in target
    if local and not Path(target).exists():
        parser.error(f"{target} not found. See docs/DATA.md for how to get it.")

    if target.endswith(".seg"):
        return run_segment(target, args)
    if local and Path(target).is_file():
        return run_csv(target, args)
    return run_index(target, args)


if __name__ == "__main__":
    raise SystemExit(main())
