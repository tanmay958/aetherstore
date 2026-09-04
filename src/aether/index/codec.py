"""Integer compression for posting lists.

Almost every byte of an inverted index is a document id. A month of REES46 is
roughly 40M events at ~8 terms each, so on the order of 300M document ids get
written down. At a fixed four bytes apiece that is 1.2 GB of pure identifiers,
which is why how they are written is not a detail.

Two ideas, applied in order, and each one depends on the previous stage of
this project having got something right.

**Delta encoding.** Posting lists are ascending, because document ids are
handed out in increasing order as events arrive. So instead of the values,
store the gaps:

    [1000, 1003, 1007, 1100]  ->  [1000, 3, 4, 93]

Same information, far smaller numbers. This only works because the list is
sorted, which is the invariant PostingList.append has been enforcing since
step 1.

**Variable-length integers.** Small numbers should not cost four bytes. A
varint spends seven bits of each byte on the value and the eighth as a "there
is more" flag, so 0-127 costs one byte, 128-16,383 costs two, and so on. After
delta encoding most gaps are tiny, so most ids collapse to a single byte.

Together that is roughly 4x on the ids and rather more on dense lists, where a
common term appearing in consecutive documents produces a run of gaps equal to
1.

What this is not: a production codec. A real one bit-packs blocks of 128
values at the width the largest in the block requires, using vectorized
numpy shifts, which is both smaller and considerably faster to decode than
byte-at-a-time varints in Python. The structure here, separate runs for ids
and frequencies, is arranged so that swapping the encoding of a run does not
disturb anything else.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def encode_varint(value: int) -> bytes:
    """One non-negative integer, seven bits per byte, high bit as continuation."""
    if value < 0:
        raise ValueError(f"varints are unsigned, got {value}")
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def encode_varints(values: Iterable[int]) -> bytes:
    out = bytearray()
    for value in values:
        if value < 0:
            raise ValueError(f"varints are unsigned, got {value}")
        while value >= 0x80:
            out.append((value & 0x7F) | 0x80)
            value >>= 7
        out.append(value)
    return bytes(out)


def decode_varint(data: bytes, pos: int = 0) -> tuple[int, int]:
    """Read one varint. Returns the value and the position after it."""
    value = 0
    shift = 0
    while True:
        try:
            byte = data[pos]
        except IndexError:
            raise ValueError("truncated varint") from None
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7


def decode_varints(data: bytes, count: int, pos: int = 0) -> tuple[list[int], int]:
    """Read `count` varints. Returns the values and the position after them."""
    values = []
    for _ in range(count):
        value = 0
        shift = 0
        while True:
            try:
                byte = data[pos]
            except IndexError:
                raise ValueError("truncated varint") from None
            pos += 1
            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
        values.append(value)
    return values, pos


def delta_encode(values: Sequence[int]) -> list[int]:
    """Turn an ascending sequence into its gaps.

    The first value is kept whole; every later one becomes the distance from
    its predecessor. Requires strictly ascending input, which posting lists
    guarantee, and which is checked because a violation would encode a
    negative gap and corrupt the list silently.

        >>> delta_encode([1000, 1003, 1007, 1100])
        [1000, 3, 4, 93]
    """
    if not values:
        return []
    gaps = [values[0]]
    previous = values[0]
    for value in values[1:]:
        if value <= previous:
            raise ValueError(
                f"delta encoding needs ascending values: {value} follows {previous}"
            )
        gaps.append(value - previous)
        previous = value
    return gaps


def delta_decode(gaps: Sequence[int]) -> list[int]:
    """Rebuild the original values from their gaps."""
    values = []
    running = 0
    for gap in gaps:
        running += gap
        values.append(running)
    return values
