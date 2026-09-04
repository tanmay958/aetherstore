"""Verify that a store actually works, before trusting an index to it.

Exercises exactly the operations the segment reader depends on, in the order
it depends on them, and times each. A misconfigured bucket, a bad credential,
or an S3 clone that quietly mishandles suffix ranges shows up here in a few
seconds instead of halfway through indexing forty million events.

    python -m aether.storage.check data/segments
    python -m aether.storage.check s3://aether            # MinIO or AWS
    python -m aether.storage.check r2://aether            # Cloudflare R2

The timings are the interesting part. Locally every read is a seek and lands
around a tenth of a millisecond; against R2 or S3 the same reads take tens of
milliseconds each. That gap is the entire justification for the segment
format, and this is the cheapest way to see it on your own connection.
"""

from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

from aether.storage import open_store
from aether.storage.base import ObjectStore
from aether.env import load_dotenv

# Big enough that a suffix read is unambiguous, small enough to be free.
PAYLOAD = bytes(range(256)) * 16  # 4 KiB
FOOTER_LEN = 116  # the real segment footer size, so the check mirrors reality


def _timed(label: str, fn) -> tuple[str, float, object]:
    started = time.perf_counter()
    value = fn()
    return label, (time.perf_counter() - started) * 1000, value


def check(store: ObjectStore, key: str) -> list[tuple[str, float, bool, str]]:
    """Run the operations a segment reader needs. Returns per-step results."""
    results: list[tuple[str, float, bool, str]] = []

    def record(label: str, fn, verify) -> None:
        try:
            _, ms, value = _timed(label, fn)
        except Exception as exc:  # surfaced, not swallowed
            results.append((label, 0.0, False, f"{type(exc).__name__}: {exc}"))
            raise
        ok, note = verify(value)
        results.append((label, ms, ok, note))

    record("put 4 KiB", lambda: store.put(key, PAYLOAD), lambda _: (True, ""))
    record(
        "size",
        lambda: store.size(key),
        lambda n: (n == len(PAYLOAD), f"{n} bytes"),
    )
    record("exists", lambda: store.exists(key), lambda ok: (ok is True, ""))
    record(
        "get_range middle",
        lambda: store.get_range(key, 1000, 64),
        # The inclusive-range off-by-one shows up here and nowhere else.
        lambda b: (b == PAYLOAD[1000:1064], f"{len(b)} bytes, want 64"),
    )
    record(
        "get_suffix footer",
        lambda: store.get_suffix(key, FOOTER_LEN),
        lambda b: (b == PAYLOAD[-FOOTER_LEN:], f"{len(b)} bytes, want {FOOTER_LEN}"),
    )
    record(
        "get_range past end",
        lambda: store.get_range(key, len(PAYLOAD) + 10, 32),
        lambda b: (b == b"", "empty, as expected"),
    )
    record(
        "get_range whole",
        lambda: store.get_range(key, 0, len(PAYLOAD)),
        lambda b: (b == PAYLOAD, f"{len(b)} bytes"),
    )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aether.storage.check",
        description="Verify a store supports everything the segment reader needs.",
    )
    parser.add_argument("uri", help="a path, s3://bucket, or r2://bucket")
    parser.add_argument(
        "--keep", action="store_true", help="leave the probe object behind"
    )
    args = parser.parse_args(argv)
    load_dotenv()

    try:
        store = open_store(args.uri)
    except (ValueError, ImportError) as exc:
        print(f"cannot open {args.uri}: {exc}")
        return 2

    key = f"_aether_check_{uuid.uuid4().hex[:12]}"
    print(f"checking {args.uri}  ({type(store).__name__})")
    endpoint = getattr(store, "endpoint_url", None)
    if endpoint:
        print(f"  endpoint {endpoint}  region {getattr(store, 'region', '?')}")
    print()

    try:
        results = check(store, key)
    except Exception:
        # The failing step is already recorded and printed below by the
        # caller's own report, so re-raising here would only add noise.
        print("  FAILED. The most common causes:")
        print("    - credentials missing or wrong (AWS_ACCESS_KEY_ID/SECRET)")
        print("    - bucket does not exist yet")
        print("    - wrong endpoint; see docs/STORAGE.md")
        raise SystemExit(1) from None

    ok = True
    for label, ms, passed, note in results:
        mark = "ok  " if passed else "FAIL"
        ok = ok and passed
        print(f"  {mark} {label:<20} {ms:>8.2f} ms   {note}")

    if not args.keep:
        store.delete(key)
        print(f"\n  cleaned up {key!r}")

    total = sum(ms for _, ms, _, _ in results)
    print(f"\n  {'all checks passed' if ok else 'CHECKS FAILED'}   {total:.1f} ms total")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
