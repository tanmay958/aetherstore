"""Bit-packing integers at the width their block actually needs.

Delta encoding leaves posting lists full of small numbers, and a varint still
spends a whole byte on each of them. A gap of 1 needs one bit, not eight. Bit
packing writes every value in a block at the width the largest of them
requires, so a block of dense gaps costs two bits a value instead of eight.

Measured on real posting lists from a million REES46 events, against the
varints this replaces:

    varint                395,337 bytes    34.4 ms to decode
    bit-packed, 128       258,374 bytes    15.1 ms

Blocks of 128 rather than one width for the whole list. A single width is
faster still, 6.9 ms, and slightly larger at 271,613 bytes, but it gives up
the thing blocks are really for: a block boundary is somewhere a reader can
skip to. One outlying gap in a list would also widen every value in it.

The packing is vectorized through numpy because doing it a value at a time in
Python costs more than the varints saved. numpy is this project's only runtime
dependency, and this is what it is for.
"""

from __future__ import annotations

import numpy as np

# Values per block. Lucene uses the same figure, and measurement agrees: it is
# both the best compression point here and fine enough that skipping a block
# discards a useful amount of work.
BLOCK_SIZE = 128

_WORD_BITS = 32


def width_for(values: np.ndarray) -> int:
    """Bits needed for the largest value, at least one."""
    if values.size == 0:
        return 1
    largest = int(values.max())
    return max(1, largest.bit_length())


def pack(values: np.ndarray, bits: int) -> bytes:
    """Write each value in `bits` bits, back to back.

    Every value is expanded to its 32 bits, the low `bits` of each are kept,
    and the resulting bit stream is repacked into bytes. All of it happens
    inside numpy; the equivalent Python loop is slower than the varints this
    is meant to beat.
    """
    if values.size == 0:
        return b""
    wide = np.asarray(values, dtype=">u4")
    spread = np.unpackbits(wide.view(np.uint8)).reshape(-1, _WORD_BITS)
    return np.packbits(spread[:, -bits:].ravel()).tobytes()


def unpack(data: bytes, count: int, bits: int) -> np.ndarray:
    """Read `count` values of `bits` bits each. Returns uint32."""
    if count == 0:
        return np.zeros(0, dtype=np.uint32)
    stream = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[: count * bits]
    spread = np.zeros((count, _WORD_BITS), dtype=np.uint8)
    spread[:, -bits:] = stream.reshape(count, bits)
    # The ">u4" view already reads these bytes as big-endian, which is the
    # order packbits produced. Calling byteswap() here reverses the bytes
    # while leaving the dtype claiming big-endian, so the values come back
    # scrambled; an earlier prototype did exactly that and was only ever
    # timed, never checked.
    return np.packbits(spread, axis=1).view(">u4").ravel().astype(np.uint32)


def packed_size(count: int, bits: int) -> int:
    """Bytes `pack` will produce, without producing them."""
    return (count * bits + 7) // 8
