"""Compact an index, and optionally collect what compaction retires.

    python -m aether.index.compact data/idx
    python -m aether.index.compact r2://aether/idx --merge-factor 10 --gc --delete

A command rather than a background thread inside the indexer. Lucene and
RocksDB compact continuously in-process, which is right at scale, but it would
put a second writer on a manifest whose safety currently comes from Kafka
guaranteeing exactly one consumer per partition. That guarantee is the reason
this engine needs no lock and no consensus algorithm, and giving it up to save
running a command is a poor trade. A scheduled job can call this after
indexing.
"""

from __future__ import annotations

import argparse

from aether.env import load_dotenv
from aether.index.compactor import (
    DEFAULT_MAX_MERGED_BYTES,
    DEFAULT_MERGE_FACTOR,
    compact,
    plan_compaction,
)
from aether.index.gc import DEFAULT_GRACE_SECONDS, collect
from aether.index.manifest import DEFAULT_MANIFEST_KEY, read_manifest
from aether.storage import CountingStore, open_store


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aether.index.compact",
        description="Merge small segments, and collect what no manifest references.",
    )
    parser.add_argument("store", help="a directory, s3://bucket/prefix, or r2://...")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_KEY, help="manifest key")
    parser.add_argument(
        "--merge-factor",
        type=int,
        default=DEFAULT_MERGE_FACTOR,
        help=f"segments per merge (default: {DEFAULT_MERGE_FACTOR})",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_MERGED_BYTES,
        help="ceiling on a merged segment",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show the plan without merging"
    )
    parser.add_argument(
        "--gc", action="store_true", help="also look for unreferenced objects"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="with --gc, actually remove them. Without this the collector only reports.",
    )
    parser.add_argument(
        "--grace",
        type=float,
        default=DEFAULT_GRACE_SECONDS,
        help="leave objects younger than this alone, in seconds "
        f"(default: {DEFAULT_GRACE_SECONDS:.0f})",
    )
    args = parser.parse_args(argv)
    load_dotenv()

    store = CountingStore(open_store(args.store))
    manifest = read_manifest(store, args.manifest)

    if not manifest.segments:
        print(f"no manifest at {args.store}. Build one with aether.index.ingest.")
        return 1

    print(f"index {args.store}")
    print(
        f"  {len(manifest.segments):,} segments, {manifest.docs:,} documents, "
        f"{human(manifest.bytes)}"
    )
    # Two requests per segment is what opening the index costs any cold
    # process, and it is the number compaction exists to reduce.
    print(f"  opening costs {2 * len(manifest.segments) + 1:,} requests")
    print()

    plan = plan_compaction(
        manifest, merge_factor=args.merge_factor, max_merged_bytes=args.max_bytes
    )

    if not plan:
        print(f"  nothing to merge at --merge-factor {args.merge_factor}")
    else:
        print(f"plan: {plan.segments_in} segments -> {plan.segments_out}")
        for group in plan.groups:
            print(
                f"    {len(group):>3} segments  {human(sum(s.bytes for s in group)):>10}"
                f"  {sum(s.docs for s in group):>9,} docs"
            )
        print()

        if args.dry_run:
            print("  --dry-run, nothing written")
        else:
            result = compact(
                store,
                manifest_key=args.manifest,
                merge_factor=args.merge_factor,
                max_merged_bytes=args.max_bytes,
                on_merge=lambda group, merged: print(
                    f"    merged {len(group)} -> {merged.key}  "
                    f"{merged.docs:,} docs  {human(merged.bytes)}"
                ),
            )
            after = read_manifest(store, args.manifest)
            print()
            print(f"  {result}")
            print(
                f"  segments {len(manifest.segments):,} -> {len(after.segments):,}, "
                f"opening {2 * len(manifest.segments) + 1:,} -> "
                f"{2 * len(after.segments) + 1:,} requests"
            )
            print(
                "  sources left in place for in-flight readers; "
                "use --gc --delete once they are past the grace period"
            )

    if args.gc:
        print()
        outcome = collect(
            store, grace_seconds=args.grace, delete=args.delete
        )
        print(f"collector ({len(outcome.manifests)} manifests)")
        print(f"  {outcome}")
        for orphan in outcome.deleted[:10]:
            verb = "deleted" if args.delete else "would delete"
            print(f"    {verb} {orphan.key}  {human(orphan.bytes)}")
        if len(outcome.deleted) > 10:
            print(f"    ... and {len(outcome.deleted) - 10} more")
        if outcome.spared:
            print(
                f"    spared {len(outcome.spared)} younger than "
                f"{args.grace:.0f}s, which may be mid-publish or still being read"
            )
        if outcome.deleted and not args.delete:
            print("  --delete to remove them")

    print()
    print(f"  storage cost   {store.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
