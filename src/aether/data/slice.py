"""Carve a small, self-contained slice out of the full REES46 CSV.

A full month is 5.5 GB expanded, which is the wrong thing to re-read every
time you change a line of the segment encoder. This produces a slice measured
in megabytes that iterates in seconds.

The output is a valid REES46 CSV with the same header, so every downstream
tool reads it through exactly the same code path as the full file. No special
cases, no "test mode".

It slices by whole sessions rather than by row count. A session cut in half
has a broken funnel: it can show an add_to_cart whose matching purchase was
truncated away, which turns a converted session into a fake abandonment and
poisons the label the ML phase depends on.

    python -m aether.data.slice --input data/raw/2019-Oct.csv.gz \\
        --sessions 2000 --out data/slices/dev.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from aether.data.rees46 import REES46_COLUMNS, _resolve_columns, open_csv

# Sessions are chosen from only the first slice of the scan window so that the
# rest of the window can still catch their later events. Picking a session
# that first appears near the end of the window would guarantee truncating it.
_SELECTION_FRACTION = 5


@dataclass
class SliceStats:
    rows_scanned: int = 0
    sessions_selected: int = 0
    rows_written: int = 0
    bytes_written: int = 0
    truncated_warning: bool = False


def _select_sessions(path: Path, wanted: int, scan_limit: int) -> tuple[set[str], int]:
    """First pass: collect the first `wanted` distinct session ids.

    Only the opening portion of the scan window is considered, so that pass
    two still has room to pick up the tail of each chosen session.
    """
    selection_limit = max(1, scan_limit // _SELECTION_FRACTION)
    sessions: set[str] = set()
    rows = 0

    with open_csv(path) as handle:
        reader = csv.reader(handle)
        header = next(reader)
        i_session = _resolve_columns(header, path)["user_session"]

        for row in reader:
            rows += 1
            if rows > selection_limit or len(sessions) >= wanted:
                break
            if len(row) > i_session and row[i_session]:
                sessions.add(row[i_session])

    return sessions, rows


def slice_sessions(
    src: Path, dst: Path, *, sessions: int = 2000, scan_limit: int = 2_000_000
) -> SliceStats:
    """Write every row belonging to the first `sessions` distinct sessions.

    Two passes over a bounded prefix of the file, so the cost is capped by
    `scan_limit` rather than by the size of the input.

    A session whose events extend past `scan_limit` is still truncated. Raising
    `scan_limit` shrinks that risk; the returned stats flag when the window was
    fully consumed, which is when truncation becomes possible.
    """
    if sessions < 1:
        raise ValueError("sessions must be at least 1")

    selected, _ = _select_sessions(src, sessions, scan_limit)
    stats = SliceStats(sessions_selected=len(selected))

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open_csv(src) as handle, dst.open("w", encoding="utf-8", newline="") as out:
        reader = csv.reader(handle)
        header = next(reader)
        i_session = _resolve_columns(header, src)["user_session"]

        writer = csv.writer(out)
        writer.writerow(header)

        for row in reader:
            stats.rows_scanned += 1
            if stats.rows_scanned > scan_limit:
                # The window ran out before the file did, so a selected session
                # may continue past this point and be cut short.
                stats.truncated_warning = True
                break
            if len(row) > i_session and row[i_session] in selected:
                writer.writerow(row)
                stats.rows_written += 1

    stats.bytes_written = dst.stat().st_size
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aether.data.slice",
        description="Carve a whole-session slice out of a REES46 CSV.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/2019-Oct.csv.gz"),
        help="full REES46 .csv or .csv.gz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/slices/dev.csv"),
        help="where to write the slice",
    )
    parser.add_argument(
        "--sessions", type=int, default=2000, help="how many whole sessions to keep"
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=2_000_000,
        help="how many input rows to scan (bounds the work)",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(f"{args.input} not found. See docs/DATA.md for how to get it.")

    stats = slice_sessions(
        args.input, args.out, sessions=args.sessions, scan_limit=args.scan_limit
    )

    print(f"wrote {args.out}")
    print(f"  sessions       {stats.sessions_selected:,}")
    print(f"  rows scanned   {stats.rows_scanned:,}")
    print(f"  rows written   {stats.rows_written:,}")
    print(f"  size           {stats.bytes_written / 1_048_576:.2f} MB")
    if stats.truncated_warning:
        print(
            "  note: hit --scan-limit, so trailing events of some sessions may be\n"
            "        missing. Raise --scan-limit if session completeness matters."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
