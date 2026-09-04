"""A bloom filter over a segment's terms.

Answers one question, in memory, for no requests at all:

    could this term be in this segment?

The answer is "definitely not" or "maybe". It is never wrong about absence,
which is the only guarantee needed: a "definitely not" skips the segment
entirely, and a false "maybe" costs one wasted dictionary read and never a
wrong result.

The benchmark is what justified building this. Searching for a term no segment
contains costs one dictionary read per segment, a hundred of them across a
million documents, purely to discover that it is not there. Rare terms are the
common case in real query logs, and against object storage each of those reads
is a billed round trip. With a filter riding along in the hotcache, which is
already fetched once and cached forever, the same query costs nothing.

This was deliberately not built earlier. On a 27-document fixture with one
segment there is nothing to skip, so there was no way to tell whether it
helped. It became obviously worth it the moment there were a hundred segments.

## How it works

A row of bits, all zero. Adding a term sets `k` of them, chosen by hashing.
Testing a term checks those same `k` bits: if any is zero the term was
certainly never added, because adding it would have set that bit. If all are
one it was probably added, or a handful of other terms happened to set exactly
those bits between them.

The `k` positions come from two base hashes combined as `h1 + i * h2`, the
Kirsch-Mitzenmacher construction, which costs two hashes instead of seven at
no measurable loss in accuracy.

Both halves come from one BLAKE2b digest, split. The first attempt used CRC32
with two different seeds, which measured a 1.6% false positive rate against a
1.0% target: CRC32 outputs for the same input under different initial values
are linearly related, so the two "independent" hashes were nothing of the
sort, and the derived sequence collapsed onto correlated positions. Splitting
one wide digest gives genuinely independent halves and hits the target.

Not the builtin `hash()`, for the same reason as everywhere else in this
project: it is salted per process, so a filter built in one run would answer
differently in the next.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Iterable

# One percent of absent terms will cost a wasted dictionary read. Tightening
# it costs hotcache, which is held in memory for the life of every reader, and
# the wasted read is cheap. Nine point six bits per term at this rate.
DEFAULT_FALSE_POSITIVE_RATE = 0.01

_HEADER = struct.Struct("<IB")  # bit count, hash count


def optimal_size(expected_terms: int, false_positive_rate: float) -> tuple[int, int]:
    """Bits and hash count for a target false positive rate.

        m = -n ln(p) / (ln 2)^2
        k = (m / n) ln 2

    Both are the standard closed forms. `k` is clamped to at least one so a
    degenerate configuration still produces a usable filter.
    """
    if expected_terms <= 0:
        return 8, 1
    if not 0.0 < false_positive_rate < 1.0:
        raise ValueError(f"false positive rate must be in (0, 1), got {false_positive_rate}")

    bits = int(math.ceil(-expected_terms * math.log(false_positive_rate) / (math.log(2) ** 2)))
    bits = max(8, bits + (-bits % 8))  # whole bytes
    hashes = max(1, int(round((bits / expected_terms) * math.log(2))))
    return bits, hashes


class BloomFilter:
    """Membership testing that may say "maybe" but never a wrong "no"."""

    __slots__ = ("bits", "hashes", "_words")

    def __init__(self, bits: int, hashes: int, words: bytearray | None = None) -> None:
        if bits % 8:
            raise ValueError(f"bit count must be a whole number of bytes, got {bits}")
        self.bits = bits
        self.hashes = hashes
        self._words = words if words is not None else bytearray(bits // 8)

    @classmethod
    def for_terms(
        cls, terms: Iterable[str], false_positive_rate: float = DEFAULT_FALSE_POSITIVE_RATE
    ) -> BloomFilter:
        terms = list(terms)
        bits, hashes = optimal_size(len(terms), false_positive_rate)
        filter_ = cls(bits, hashes)
        for term in terms:
            filter_.add(term)
        return filter_

    def _positions(self, term: str):
        # One 64-bit digest split into two independent 32-bit halves. Two
        # CRC32s with different seeds would be cheaper and are linearly
        # related, which measurably worsened the false positive rate.
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        first = int.from_bytes(digest[:4], "little")
        # Forced odd so the derived sequence strides through the whole filter
        # instead of revisiting a short cycle of positions.
        second = int.from_bytes(digest[4:], "little") | 1
        for i in range(self.hashes):
            yield (first + i * second) % self.bits

    def add(self, term: str) -> None:
        for position in self._positions(term):
            self._words[position >> 3] |= 1 << (position & 7)

    def __contains__(self, term: str) -> bool:
        """False means certainly absent. True means probably present."""
        for position in self._positions(term):
            if not self._words[position >> 3] & (1 << (position & 7)):
                return False
        return True

    def __len__(self) -> int:
        """Size in bytes, which is what it costs in the hotcache."""
        return _HEADER.size + len(self._words)

    @property
    def load(self) -> float:
        """Fraction of bits set. Far above one half means the filter is too
        small for what was put in it and false positives will be common."""
        return sum(bin(byte).count("1") for byte in self._words) / self.bits

    # -- serialization -----------------------------------------------------

    def to_bytes(self) -> bytes:
        return _HEADER.pack(self.bits, self.hashes) + bytes(self._words)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> tuple[BloomFilter, int]:
        bits, hashes = _HEADER.unpack_from(data, offset)
        start = offset + _HEADER.size
        end = start + bits // 8
        return cls(bits, hashes, bytearray(data[start:end])), end
