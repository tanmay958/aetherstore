"""Posting lists in skippable, bit-packed blocks.

Two optimizations that turn out to be one structure. Bit packing wants fixed
size blocks so each can use the width its own values need. Skipping wants
somewhere to jump to. A block boundary is both.

Layout for one term:

    varint  count
    varint  block count
    [skip index]  per block:  varint last doc id delta
                              varint byte length
                              u8    id width
                              u8    frequency width
    [data]        per block:  packed id gaps, then packed frequencies

The skip index is small, read first, and answers "which block could contain
document N" without decoding any of the data. That is what makes an
intersection cheap when one side is selective: pairing a term in 100 documents
with one in 89,000 previously walked all 89,000, and now touches only the
blocks that could hold a match.

Short lists take a different path entirely. The median term in real REES46
data appears in two documents and 98.7% of terms fit in a single block, so a
skip index and two numpy calls for a list of one document is all overhead:
numpy on an array that small is dominated by the call itself, and there is
nothing to skip within a block. Lists at or below `SMALL_LIST` are plain
varints, and the count in the header says which encoding follows.

Skipping that path entirely was measured and rejected: indexing dropped from
62,000 documents a second to 45,000, because 98.7% of terms were paying numpy
call overhead to save nothing.

Decoding is lazy and cached. A block decoded for one query term stays decoded
for the rest of that query.

The compression figures, measured against the varints this replaces on real
posting lists from a million events:

    varint                  395,337 bytes    34.4 ms
    bit-packed, 128         258,374 bytes    15.1 ms

A single width for a whole list decodes faster still, 6.9 ms, but has no block
boundaries to skip to and lets one outlying gap widen every value in the list.
"""

from __future__ import annotations

from bisect import bisect_left

import numpy as np

from aether.index.bitpack import BLOCK_SIZE, pack, unpack, width_for
from aether.index.codec import (
    decode_varint,
    decode_varints,
    delta_decode,
    delta_encode,
    encode_varint,
    encode_varints,
)

# At or below this many documents, a list is varint encoded with no blocks and
# no skip index. The crossover was measured on identical lists rather than
# guessed: a skip index plus two bit-packed runs costs 19 bytes for eight
# documents where varints cost 17, and blocking only starts winning around
# twelve. There is nothing to skip inside a single block either way.
SMALL_LIST = 8


def encode_blocked(doc_ids: list[int], freqs: list[int]) -> bytes:
    """Encode one posting list. Doc ids must be ascending."""
    count = len(doc_ids)
    if count == 0:
        return encode_varint(0)

    if count <= SMALL_LIST:
        # No numpy, no blocks, no skip index. This is the overwhelming
        # majority of terms and the reason indexing is not slower than it was.
        return (
            encode_varint(count)
            + encode_varints(delta_encode(doc_ids))
            + encode_varints(freqs)
        )

    ids = np.asarray(doc_ids, dtype=np.int64)
    # Gaps, not values. Ascending ids make these small, which is the whole
    # reason a narrow bit width is available at all.
    gaps = np.diff(ids, prepend=0).astype(np.uint32)
    frequencies = np.asarray(freqs, dtype=np.uint32)

    skip: list[tuple[int, int, int, int]] = []
    payload = bytearray()
    previous_last = 0

    for start in range(0, count, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, count)
        gap_block = gaps[start:end]
        freq_block = frequencies[start:end]
        id_bits = width_for(gap_block)
        freq_bits = width_for(freq_block)

        block = pack(gap_block, id_bits) + pack(freq_block, freq_bits)
        last_id = int(ids[end - 1])
        skip.append((last_id - previous_last, len(block), id_bits, freq_bits))
        previous_last = last_id
        payload += block

    out = bytearray(encode_varint(count))
    out += encode_varint(len(skip))
    for last_delta, length, id_bits, freq_bits in skip:
        out += encode_varint(last_delta)
        out += encode_varint(length)
        out += bytes((id_bits, freq_bits))
    out += payload
    return bytes(out)


class BlockedPostings:
    """A posting list read a block at a time.

    Presents the same surface as `PostingList`, so callers that only want the
    whole thing are unaffected, while an intersection that can skip gets to
    ask which blocks matter first.
    """

    __slots__ = (
        "count", "_last_ids", "_offsets", "_lengths", "_id_bits",
        "_freq_bits", "_data", "_cache", "_small",
    )

    def __init__(self, data: bytes) -> None:
        count, pos = decode_varint(data)
        self.count = count
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        self._last_ids: list[int] = []
        self._offsets: list[int] = []
        self._lengths: list[int] = []
        self._id_bits: list[int] = []
        self._freq_bits: list[int] = []
        self._data = b""
        self._small: tuple[list[int], list[int]] | None = None

        if count == 0:
            return

        if count <= SMALL_LIST:
            gaps, pos = decode_varints(data, count, pos)
            freqs, _ = decode_varints(data, count, pos)
            ids = delta_decode(gaps)
            self._small = (ids, freqs)
            # One notional block, so the skip index still answers questions
            # about this list without every caller special-casing it.
            self._last_ids = [ids[-1]]
            self._cache[0] = (np.asarray(ids, np.int64), np.asarray(freqs, np.int64))
            return

        block_count, pos = decode_varint(data, pos)
        last_ids: list[int] = []
        lengths: list[int] = []
        id_bits: list[int] = []
        freq_bits: list[int] = []
        running = 0
        for _ in range(block_count):
            delta, pos = decode_varint(data, pos)
            length, pos = decode_varint(data, pos)
            running += delta
            last_ids.append(running)
            lengths.append(length)
            id_bits.append(data[pos])
            freq_bits.append(data[pos + 1])
            pos += 2

        offsets, running_offset = [], 0
        for length in lengths:
            offsets.append(running_offset)
            running_offset += length

        self._last_ids = last_ids
        self._offsets = offsets
        self._lengths = lengths
        self._id_bits = id_bits
        self._freq_bits = freq_bits
        self._data = data[pos:]

    # -- shape -------------------------------------------------------------

    @property
    def df(self) -> int:
        return self.count

    @property
    def block_count(self) -> int:
        return len(self._last_ids)

    @property
    def last_doc_id(self) -> int:
        return self._last_ids[-1] if self._last_ids else -1

    def __len__(self) -> int:
        return self.count

    # -- blocks ------------------------------------------------------------

    def block_containing(self, target: int) -> int:
        """Index of the first block that could hold `target`.

        A binary search over the skip index, in memory, touching none of the
        packed data. Returns `block_count` when every block ends before the
        target, which means the list is exhausted.
        """
        return bisect_left(self._last_ids, target)

    def decode_block(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Absolute doc ids and frequencies for one block."""
        cached = self._cache.get(index)
        if cached is not None:
            return cached

        start = self._offsets[index]
        size = self._lengths[index]
        raw = self._data[start : start + size]

        # How many values this block holds: full, except possibly the last.
        first = index * BLOCK_SIZE
        n = min(BLOCK_SIZE, self.count - first)

        id_bits = self._id_bits[index]
        id_bytes = (n * id_bits + 7) // 8
        gaps = unpack(raw[:id_bytes], n, id_bits)
        freqs = unpack(raw[id_bytes:], n, self._freq_bits[index])

        # Gaps are relative to the previous block's last id.
        base = self._last_ids[index - 1] if index else 0
        ids = np.cumsum(gaps.astype(np.int64)) + base

        decoded = (ids, freqs.astype(np.int64))
        self._cache[index] = decoded
        return decoded

    # -- whole-list access -------------------------------------------------

    @property
    def doc_ids(self) -> list[int]:
        """Every doc id. Decodes the entire list, which skipping exists to
        avoid, so it is for callers that genuinely need all of it."""
        if self.count == 0:
            return []
        if self._small is not None:
            return self._small[0]
        return np.concatenate(
            [self.decode_block(i)[0] for i in range(self.block_count)]
        ).tolist()

    @property
    def freqs(self) -> list[int]:
        if self.count == 0:
            return []
        if self._small is not None:
            return self._small[1]
        return np.concatenate(
            [self.decode_block(i)[1] for i in range(self.block_count)]
        ).tolist()

    def freq_in(self, doc_id: int) -> int:
        index = self.block_containing(doc_id)
        if index >= self.block_count:
            return 0
        ids, freqs = self.decode_block(index)
        position = int(np.searchsorted(ids, doc_id))
        if position < len(ids) and ids[position] == doc_id:
            return int(freqs[position])
        return 0


def intersect_skipping(candidates: list[int], postings: BlockedPostings) -> list[int]:
    """Documents in both, decoding only the blocks that could contain a match.

    `candidates` is the shorter, already-decoded side. For each one, the skip
    index names the single block that could hold it, so a selective term
    paired with a common one no longer walks the common term's whole list.

    Blocks are cached, so a run of candidates falling in the same block costs
    one decode between them.
    """
    if not candidates or postings.count == 0:
        return []

    out: list[int] = []
    block_count = postings.block_count
    index = 0
    ids = freqs = None

    for candidate in candidates:
        if candidate > postings.last_doc_id:
            break
        target = postings.block_containing(candidate)
        if target >= block_count:
            break
        if ids is None or target != index:
            index = target
            ids, _ = postings.decode_block(index)
        position = int(np.searchsorted(ids, candidate))
        if position < len(ids) and ids[position] == candidate:
            out.append(candidate)
    return out
