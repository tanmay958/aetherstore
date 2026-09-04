"""Streaming loader for the REES46 public e-commerce clickstream.

Dataset: "eCommerce behavior data from multi category store", roughly 285M
events across seven months. See docs/DATA.md for download and attribution.

Published columns:

    event_time     "2019-10-01 00:00:00 UTC"
    event_type     view | cart | remove_from_cart | purchase
    product_id     numeric id
    category_id    numeric id, not indexed
    category_code  dotted taxonomy, e.g. electronics.smartphone. Often empty.
    brand          downcased. Often empty.
    price          float
    user_id        numeric id
    user_session   session uuid, changes after a long pause

Two properties drive the implementation.

It streams. One month is 1.6 GB gzipped and around 5.5 GB expanded, so nothing
here materializes rows into a list. `iter_events` is a generator and the CLI
writes each event as it is produced, so memory stays flat whether you read a
thousand rows or forty million.

It tolerates gaps. `brand` and `category_code` are missing on a large fraction
of real rows, some prices do not parse, and a few rows are malformed. Those
become None, or are skipped and counted, rather than raising. A loader that
dies on row 12,000,003 of a 40M row file is useless.

    python -m aether.data.rees46 --input data/raw/2019-Oct.csv.gz --limit 100000
"""

from __future__ import annotations

import argparse
import calendar
import csv
import gzip
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterator

from aether.data.titles import derive_title
from aether.events import make_event

# Columns as published. The loader resolves these by name from the header, so
# a reordering upstream cannot silently shift every field by one position.
REES46_COLUMNS = (
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
)

# REES46's "cart" becomes "add_to_cart", so this loader and anything added
# later both speak the vocabulary defined in aether.events.
EVENT_TYPE_MAP = {
    "view": "view",
    "cart": "add_to_cart",
    "remove_from_cart": "remove_from_cart",
    "purchase": "purchase",
}

_PROGRESS_EVERY = 1_000_000

# Epoch of midnight for every date string seen so far.
_DAY_EPOCH_CACHE: dict[str, int] = {}


def parse_timestamp(value: str) -> int:
    """Parse "2019-10-01 00:00:00 UTC" into an epoch int.

    `datetime.strptime` costs roughly 5 microseconds per call, which is over
    three minutes across a 40M row file spent on nothing but timestamps. Rows
    arrive in time order, so the date portion repeats for long runs: caching
    each day's epoch cuts the per-row work down to three int() calls and two
    multiplications.

    Raises ValueError on anything that is not this exact fixed-width format.
    """
    day = value[:10]
    base = _DAY_EPOCH_CACHE.get(day)
    if base is None:
        base = calendar.timegm(
            (int(day[0:4]), int(day[5:7]), int(day[8:10]), 0, 0, 0, 0, 0, 0)
        )
        _DAY_EPOCH_CACHE[day] = base
    return base + int(value[11:13]) * 3600 + int(value[14:16]) * 60 + int(value[17:19])


def open_csv(path: Path) -> IO[str]:
    """Open a REES46 CSV, transparently handling gzip.

    utf-8-sig strips a byte order mark if present, which would otherwise
    corrupt the first header name and break column resolution.
    """
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8-sig", newline="")
    return path.open(mode="rt", encoding="utf-8-sig", newline="")


@dataclass
class LoadStats:
    """Counters accumulated while streaming. Mutated in place by the loader."""

    rows_read: int = 0
    rows_skipped: int = 0
    events: int = 0
    event_types: Counter = field(default_factory=Counter)
    skip_reasons: Counter = field(default_factory=Counter)
    missing_brand: int = 0
    missing_category: int = 0
    missing_title: int = 0
    min_ts: int | None = None
    max_ts: int | None = None

    def observe_ts(self, ts: int) -> None:
        if self.min_ts is None or ts < self.min_ts:
            self.min_ts = ts
        if self.max_ts is None or ts > self.max_ts:
            self.max_ts = ts


def _resolve_columns(header: list[str], path: Path) -> dict[str, int]:
    try:
        return {name: header.index(name) for name in REES46_COLUMNS}
    except ValueError as exc:
        raise ValueError(
            f"{path} does not look like a REES46 CSV. "
            f"Expected columns {list(REES46_COLUMNS)}, found {header}."
        ) from exc


def iter_events(
    path: Path,
    *,
    limit: int | None = None,
    derive_titles: bool = True,
    stats: LoadStats | None = None,
    progress: bool = False,
) -> Iterator[dict]:
    """Stream a REES46 CSV as canonical events.

    A generator by design: it never holds more than one row in memory, so the
    same call works on the committed test fixture and on a 5.5 GB month.

    `event_id` is synthesized as "r_000000001" from emission order, because
    REES46 has no event identifier. It is deterministic for a given input file
    and limit, which is all any downstream code needs from it.
    """
    if stats is None:
        stats = LoadStats()

    with open_csv(path) as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return
        columns = _resolve_columns(header, path)

        # Resolved once, outside the loop. Dict lookups per row are not free
        # across tens of millions of rows.
        i_time = columns["event_time"]
        i_type = columns["event_type"]
        i_product = columns["product_id"]
        i_category = columns["category_code"]
        i_brand = columns["brand"]
        i_price = columns["price"]
        i_user = columns["user_id"]
        i_session = columns["user_session"]
        width = len(header)

        for row in reader:
            stats.rows_read += 1

            if progress and stats.rows_read % _PROGRESS_EVERY == 0:
                print(
                    f"  ... {stats.rows_read:,} rows read, {stats.events:,} events",
                    file=sys.stderr,
                )

            if len(row) != width:
                stats.rows_skipped += 1
                stats.skip_reasons["malformed row"] += 1
                continue

            event_type = EVENT_TYPE_MAP.get(row[i_type])
            if event_type is None:
                stats.rows_skipped += 1
                stats.skip_reasons[f"unknown event_type {row[i_type]!r}"] += 1
                continue

            try:
                ts = parse_timestamp(row[i_time])
            except (ValueError, IndexError):
                stats.rows_skipped += 1
                stats.skip_reasons["unparseable event_time"] += 1
                continue

            brand = row[i_brand].strip() or None
            category = row[i_category].strip() or None
            product_id = row[i_product].strip() or None

            if brand is None:
                stats.missing_brand += 1
            if category is None:
                stats.missing_category += 1

            raw_price = row[i_price].strip()
            try:
                price = float(raw_price) if raw_price else None
            except ValueError:
                price = None

            title = None
            if derive_titles and product_id:
                title = derive_title(product_id, brand, category)
            if title is None:
                stats.missing_title += 1

            stats.events += 1
            stats.event_types[event_type] += 1
            stats.observe_ts(ts)

            yield make_event(
                event_id=f"r_{stats.events:09d}",
                ts=ts,
                session_id=row[i_session],
                user_id=row[i_user],
                event_type=event_type,
                device=None,  # REES46 has no device column
                product_id=product_id,
                title=title,
                category=category,
                brand=brand,
                price=price,
                query=None,  # REES46 has no search events
            )

            if limit is not None and stats.events >= limit:
                return


def summarize(stats: LoadStats, byte_size: int | None = None) -> str:
    """Human-readable load report."""
    lines = [
        f"  rows read      {stats.rows_read:,}",
        f"  events         {stats.events:,}",
        f"  rows skipped   {stats.rows_skipped:,}",
    ]
    for reason, n in stats.skip_reasons.most_common(5):
        lines.append(f"    {reason:<34} {n:>10,}")

    if stats.min_ts is not None and stats.max_ts is not None:
        lines.append(f"  time span      {(stats.max_ts - stats.min_ts) / 3600:,.1f} h")

    if stats.events:
        pct = 100 / stats.events
        lines.append(
            f"  missing brand  {stats.missing_brand:,} ({stats.missing_brand * pct:.1f}%)"
        )
        lines.append(
            f"  missing categ. {stats.missing_category:,} ({stats.missing_category * pct:.1f}%)"
        )
        lines.append(
            f"  without title  {stats.missing_title:,} ({stats.missing_title * pct:.1f}%)"
        )
        lines.append("  event types")
        for name, n in stats.event_types.most_common():
            lines.append(f"    {name:<18} {n:>12,}  {n * pct:>5.1f}%")

    if byte_size is not None:
        lines.append(f"  written        {byte_size / 1_048_576:.1f} MB")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aether.data.rees46",
        description="Stream a REES46 clickstream CSV into canonical events.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/2019-Oct.csv.gz"),
        help="REES46 .csv or .csv.gz (default: data/raw/2019-Oct.csv.gz)",
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N events")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write JSONL here; omit to scan and report statistics only",
    )
    parser.add_argument(
        "--no-derived-titles",
        action="store_true",
        help="leave title empty instead of deriving it from product_id",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(f"{args.input} not found. See docs/DATA.md for how to get it.")

    stats = LoadStats()
    events = iter_events(
        args.input,
        limit=args.limit,
        derive_titles=not args.no_derived_titles,
        stats=stats,
        progress=True,
    )

    byte_size: int | None = None
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        byte_size = 0
        with args.out.open("w", encoding="utf-8") as handle:
            for event in events:
                line = json.dumps(event, separators=(",", ":")) + "\n"
                handle.write(line)
                byte_size += len(line)
        print(f"wrote {args.out}")
    else:
        for _ in events:
            pass
        print(f"scanned {args.input}")

    print(summarize(stats, byte_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
