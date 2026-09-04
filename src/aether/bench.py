"""Measure the engine on real data.

Every number in the README comes from here, so they can be regenerated rather
than believed. Until this existed the project's figures were extrapolations
from a 27 document fixture, which is a fine way to develop and a poor way to
make claims.

    python -m aether.bench data/raw/2019-Oct.csv --events 1000000

Two costs are reported for every query, because they answer different
questions. Milliseconds are what this machine did against local files, where a
range read is a seek. Requests are what the same query would cost against
object storage, where each one is a round trip billed individually and, from a
laptop against R2, takes roughly 200 ms regardless of size. The request count
is the number that survives the move to the cloud.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from aether.data.rees46 import iter_events
from aether.index.coordinator import Coordinator
from aether.index.ingest import ingest
from aether.index.manifest import read_manifest
from aether.index.segment import FOOTER_SIZE, SegmentReader
from aether.storage import CountingStore, LocalStore

QUERIES = [
    ("common, two terms", "samsung smartphone", {}),
    ("common, one term", "electronics", {}),
    ("rare brand", "shiseido", {}),
    ("very rare", "zzzznotpresent", {}),
    ("three terms", "samsung black smartphone", {}),
    ("any of two", "samsung apple", {"mode": "or"}),
]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


def measure_index(store: LocalStore) -> dict:
    manifest = read_manifest(store)
    sections = dict(postings=0, docstore=0, termdict=0, hotcache=0, footer=0)
    postings = 0
    for meta in manifest.segments:
        footer = SegmentReader(store, meta.key).footer
        sections["postings"] += footer.postings_length
        sections["docstore"] += footer.docstore_length
        sections["termdict"] += footer.termdict_length
        sections["hotcache"] += footer.hotcache_length
        sections["footer"] += FOOTER_SIZE
        postings += footer.num_postings
    return {"manifest": manifest, "sections": sections, "postings": postings}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aether.bench", description="Benchmark the engine.")
    parser.add_argument("input", type=Path, help="REES46 .csv or .csv.gz")
    parser.add_argument("--events", type=int, default=1_000_000)
    parser.add_argument("--docs-per-segment", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=Path("data/bench"))
    parser.add_argument("--keep", action="store_true", help="reuse an existing index")
    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(f"{args.input} not found. See docs/DATA.md.")

    # -- indexing ----------------------------------------------------------

    if not args.keep and args.out.exists():
        shutil.rmtree(args.out)
    store = LocalStore(args.out)

    if not args.keep:
        began = time.perf_counter()
        ingest(
            iter_events(args.input, limit=args.events),
            store,
            docs_per_segment=args.docs_per_segment,
        )
        index_seconds = time.perf_counter() - began
    else:
        index_seconds = float("nan")

    shape = measure_index(store)
    manifest = shape["manifest"]

    # Bytes of source CSV the indexed rows came from, for a like-for-like
    # comparison rather than against the whole file.
    source_bytes = 0
    with args.input.open("rb") as handle:
        for i, line in enumerate(handle):
            if i > args.events:
                break
            source_bytes += len(line)

    print(f"\nINDEXING  {args.input.name}")
    print(f"  documents        {manifest.docs:,}")
    print(f"  segments         {len(manifest.segments):,} "
          f"({args.docs_per_segment:,} docs each)")
    if index_seconds == index_seconds:  # not NaN
        print(f"  elapsed          {index_seconds:,.1f} s")
        print(f"  throughput       {manifest.docs / index_seconds:,.0f} docs/sec "
              f"(single threaded, pure Python)")

    print(f"\nSIZE")
    print(f"  source           {human(source_bytes)}")
    print(f"  index            {human(manifest.bytes)}   "
          f"{manifest.bytes / source_bytes:.2f}x of source")
    for name, size in shape["sections"].items():
        print(f"    {name:<12} {human(size):>12}  {100 * size / manifest.bytes:>5.1f}%")
    print(f"  postings         {shape['postings']:,}")
    print(f"  bytes/posting    {shape['sections']['postings'] / shape['postings']:.2f}"
          f"   (8.00 at fixed width)")
    print(f"  hotcache         {human(shape['sections']['hotcache'])} held in memory "
          f"to avoid reading {human(manifest.bytes)}")

    # -- opening -----------------------------------------------------------

    counting = CountingStore(store)
    coordinator = Coordinator(counting, max_workers=16)
    began = time.perf_counter()
    for meta in coordinator.manifest.segments:
        coordinator.reader(meta.key)
    open_ms = (time.perf_counter() - began) * 1000
    print(f"\nOPENING   all {len(manifest.segments)} segments")
    print(f"  {counting.stats}   {open_ms:,.0f} ms")
    print("  paid once per process; footers and hotcaches can never go stale")

    # -- querying ----------------------------------------------------------

    print(f"\nQUERIES   (warm: every segment already open)")
    print(f"  {'query':<22} {'hits':>10} {'requests':>9} {'read':>10} {'ms':>7}")
    for label, query, options in QUERIES:
        coordinator.search(query, top_k=10, **options)  # warm the dictionary
        counting.reset()
        began = time.perf_counter()
        result = coordinator.search(query, top_k=10, **options)
        elapsed = (time.perf_counter() - began) * 1000
        print(f"  {label:<22} {result.total:>10,} {counting.stats.requests:>9,} "
              f"{human(counting.stats.bytes_read):>10} {elapsed:>7,.0f}")

    # -- cold queries ------------------------------------------------------

    # Warm numbers hide what the bloom filter is for: with every dictionary
    # block already cached it never has to answer anything. Cold is the state
    # a freshly started process is in, and the one a serverless deployment is
    # always in.
    print(f"\nCOLD QUERIES   (segments open, dictionaries empty)")
    print(f"  {'query':<22} {'hits':>10} {'requests':>9} {'no bloom':>9} {'saved':>7} {'ms':>7}")
    for label, query, options in QUERIES:
        fresh = CountingStore(store)
        cold = Coordinator(fresh, max_workers=16)
        for meta in cold.manifest.segments:
            cold.reader(meta.key)
        fresh.reset()
        began = time.perf_counter()
        result = cold.search(query, top_k=10, **options)
        elapsed = (time.perf_counter() - began) * 1000

        # Every bloom rejection is a dictionary read that did not happen, so
        # the counterfactual is the requests made plus the rejections.
        skipped = sum(reader.bloom_rejections for reader in cold._readers.values())
        without = fresh.stats.requests + skipped
        saved = f"{100 * skipped / without:.0f}%" if without else "-"
        print(f"  {label:<22} {result.total:>10,} {fresh.stats.requests:>9,} "
              f"{without:>9,} {saved:>7} {elapsed:>7,.0f}")

    # -- pruning -----------------------------------------------------------

    latest = max(meta.max_ts for meta in manifest.segments)
    print(f"\nTIME PRUNING   \"samsung smartphone\"")
    for label, window in (("last hour", 3600), ("last day", 86400), ("everything", None)):
        counting.reset()
        began = time.perf_counter()
        result = coordinator.search(
            "samsung smartphone", top_k=10, start=None if window is None else latest - window
        )
        elapsed = (time.perf_counter() - began) * 1000
        print(f"  {label:<22} {result.stats.segments_searched:>3}/"
              f"{result.stats.segments_total} searched  "
              f"{counting.stats.requests:>5,} requests  {elapsed:>7,.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
