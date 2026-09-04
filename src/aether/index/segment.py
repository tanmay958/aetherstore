"""Segments: a self-contained index in one immutable file, read by byte range.

A segment is a file. What makes it useful is that it is self-contained: its
own term dictionary, its own posting lists, its own copy of the documents. A
segment can be searched with nothing else present, which is what lets any
machine search any segment independently, with no coordination.

Layout, in the order the writer produces it:

    [POSTINGS]   per-term doc id and frequency runs
    [DOCSTORE]   the original events, in blocks
    [TERMDICT]   sorted terms in blocks, each pointing into POSTINGS
    [HOTCACHE]   sparse term index, bloom filter, docstore index, doc lengths
    [FOOTER]     fixed size, at the very end

The order is forced. Each section can only be written once the one before it
exists: the dictionary needs the postings offsets, the hotcache needs the
dictionary offsets, and the footer needs all of them. You physically cannot
write the footer first, because you do not know the numbers yet.

The reader consumes them in exactly reverse order, and the footer is the hinge
that makes it possible. It is a fixed size at the very end, so it can be
fetched with "give me the last 116 bytes" against a file whose size you do not
know, and it locates everything else.

Why any of this, instead of reading the file:

    RAM read       ~100 ns    free
    local seek     ~100 us    free
    S3 range GET   20-50 ms   costs money, same price for 64 B or 8 MB

On object storage the round trip is the cost, not the bytes. So the format is
arranged to minimize requests: a fixed footer found without knowing the size,
a small hotcache read once and cached forever, and a dictionary blocked so
that finding one term costs one request instead of a binary search over the
network. A cold segment answers a single-term query in three reads, a warm one
in a single read, and the two cacheable reads never repeat because the file
can never change.

Postings are delta-encoded and varint-packed (see `codec.py`), and docstore
blocks are deflated. The term dictionary is still stored plainly; front coding
it is a later refinement, and leaving it alone keeps this step's size change
attributable to the two things that actually changed.

    python -m aether.index.build tests/fixtures/rees46_sample.csv out.seg
"""

from __future__ import annotations

import json
import struct
import zlib
from bisect import bisect_right
from pathlib import Path
from typing import Iterable, Iterator

from aether.index.base import SearchableIndex
from aether.index.bloom import BloomFilter
from aether.index.codec import (
    decode_varint,
    decode_varints,
    delta_decode,
    delta_encode,
    encode_varint,
    encode_varints,
)
from aether.index.postings import PostingList
from aether.storage.base import ObjectStore
from aether.storage.local import LocalStore

SEGMENT_MAGIC = b"ATHR"
SEGMENT_VERSION = 5

# Terms per dictionary block. Finding a term costs one request for the block
# containing it, so a larger block means fetching more bytes you will discard
# and a smaller one means a bigger hotcache. 32 keeps a block around a
# kilobyte, which is far below the size at which a range read starts costing
# measurably more.
TERMS_PER_BLOCK = 32

# Documents per docstore block. Only the handful of documents actually being
# displayed are ever fetched, so this trades read amplification against the
# size of the block index.
DOCS_PER_BLOCK = 64

_FOOTER = struct.Struct(
    "<"
    "4sI"      # magic, version
    "QQQQQQQQ" # postings, docstore, termdict, hotcache: offset and length each
    "II"       # num_docs, num_terms
    "QQ"       # num_postings, total term occurrences
    "qq"       # min_ts, max_ts
    "4s"       # magic again, so the tail alone proves this is a segment
)
FOOTER_SIZE = _FOOTER.size


class Footer:
    """The segment's table of contents.

    Small, fixed size, and immutable, so a reader fetches it once and keeps it
    forever. `min_ts` and `max_ts` are here rather than in the hotcache
    because they enable the cheapest optimization available: a query filtered
    to the last hour can discard a segment whose newest event is three days
    old without issuing a single request.
    """

    __slots__ = (
        "version", "postings_offset", "postings_length", "docstore_offset",
        "docstore_length", "termdict_offset", "termdict_length",
        "hotcache_offset", "hotcache_length", "num_docs", "num_terms",
        "num_postings", "total_length", "min_ts", "max_ts",
    )

    def __init__(self, **kwargs) -> None:
        for name in self.__slots__:
            setattr(self, name, kwargs[name])

    def pack(self) -> bytes:
        return _FOOTER.pack(
            SEGMENT_MAGIC, self.version,
            self.postings_offset, self.postings_length,
            self.docstore_offset, self.docstore_length,
            self.termdict_offset, self.termdict_length,
            self.hotcache_offset, self.hotcache_length,
            self.num_docs, self.num_terms,
            self.num_postings, self.total_length,
            self.min_ts, self.max_ts,
            SEGMENT_MAGIC,
        )

    @classmethod
    def unpack(cls, data: bytes) -> Footer:
        if len(data) < FOOTER_SIZE:
            raise ValueError(
                f"not a segment: {len(data)} bytes is shorter than the "
                f"{FOOTER_SIZE} byte footer"
            )
        fields = _FOOTER.unpack(data[-FOOTER_SIZE:])
        if fields[0] != SEGMENT_MAGIC or fields[-1] != SEGMENT_MAGIC:
            raise ValueError(
                f"not a segment: footer magic was {fields[0]!r}/{fields[-1]!r}, "
                f"expected {SEGMENT_MAGIC!r}"
            )
        if fields[1] != SEGMENT_VERSION:
            raise ValueError(
                f"segment format version {fields[1]} is not supported "
                f"(this build reads version {SEGMENT_VERSION})"
            )
        names = cls.__slots__
        return cls(**dict(zip(names, fields[1:-1])))


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------


def _encode_postings(doc_ids: list[int], freqs: list[int]) -> bytes:
    """One term's posting list: a count, then ids, then frequencies.

    Ids and frequencies stay in separate runs rather than interleaved because
    they compress differently. Ids are ascending, so their gaps are tiny and
    delta encoding collapses them; frequencies are small independent integers
    that delta encoding would only make worse. Keeping the runs apart lets
    each use the representation that suits it, and lets either be replaced
    later without disturbing the other.
    """
    return (
        encode_varint(len(doc_ids))
        + encode_varints(delta_encode(doc_ids))
        + encode_varints(freqs)
    )


def _decode_postings(data: bytes) -> PostingList:
    count, pos = decode_varint(data)
    gaps, pos = decode_varints(data, count, pos)
    freqs, _ = decode_varints(data, count, pos)
    return PostingList(doc_ids=delta_decode(gaps), freqs=freqs)


def _encode_dict_block(entries: list[tuple[str, int, int, int]]) -> bytes:
    """A run of sorted terms, each with its document frequency and the byte
    range of its posting list.

    Document frequency lives here rather than only in the postings so that
    `df(term)` costs no extra request: it is already in the block that had to
    be fetched to locate the term at all.

    Three compressions apply, and the largest is the least obvious.

    **Front coding.** The terms are sorted, so neighbours share prefixes. Each
    entry stores how many leading bytes it shares with its predecessor and
    only the remainder. Half the term bytes in a real segment are shareable.
    Sharing restarts at every block boundary, because a block must stay
    independently readable: it is fetched alone, without the block before it.

    **Derived offsets.** Posting lists are written in sorted-term order and
    laid end to end, so a term's postings begin exactly where the previous
    term's ended. The offset is therefore reconstructable rather than stored,
    which removes eight bytes an entry. What is stored is the difference from
    that prediction, which is always zero today and costs one byte, so an
    encoder that ever lays postings out differently cannot silently corrupt
    the index.

    **Variable-length integers.** Document frequencies and posting lengths are
    mostly small, and a fixed 32-bit field spends four bytes saying so.
    """
    out = bytearray(encode_varint(len(entries)))
    if not entries:
        return bytes(out)

    # The block's own starting point, from which every offset in it follows.
    out += encode_varint(entries[0][2])

    previous_term = ""
    predicted_offset = entries[0][2]
    for term, df, offset, length in entries:
        shared = 0
        limit = min(len(previous_term), len(term))
        while shared < limit and previous_term[shared] == term[shared]:
            shared += 1
        suffix = term[shared:].encode("utf-8")

        out += encode_varint(shared)
        out += encode_varint(len(suffix))
        out += suffix
        out += encode_varint(df)
        out += encode_varint(offset - predicted_offset)
        out += encode_varint(length)

        previous_term = term
        predicted_offset = offset + length
    return bytes(out)


def _decode_dict_block(data: bytes) -> dict[str, tuple[int, int, int]]:
    count, pos = decode_varint(data)
    if not count:
        return {}
    block_offset, pos = decode_varint(data, pos)

    entries: dict[str, tuple[int, int, int]] = {}
    previous_term = ""
    predicted_offset = block_offset
    for _ in range(count):
        shared, pos = decode_varint(data, pos)
        suffix_length, pos = decode_varint(data, pos)
        term = previous_term[:shared] + data[pos : pos + suffix_length].decode("utf-8")
        pos += suffix_length

        df, pos = decode_varint(data, pos)
        gap, pos = decode_varint(data, pos)
        length, pos = decode_varint(data, pos)

        offset = predicted_offset + gap
        entries[term] = (df, offset, length)
        previous_term = term
        predicted_offset = offset + length
    return entries


def _encode_hotcache(
    term_blocks: list[tuple[str, int, int]],
    doc_blocks: list[tuple[int, int, int]],
    doc_lengths: list[int],
    bloom: BloomFilter,
) -> bytes:
    """Everything needed to plan a query without touching the network again.

    Once this is in memory a reader knows which dictionary block holds any
    term, which docstore block holds any document, and how long every document
    is. That planning ability is what makes coalescing and parallel fan-out
    possible later: you can only merge and parallelize requests you already
    know you need.

    Document lengths are here because BM25 needs one per candidate document,
    and a query can produce thousands of candidates. Fetching them from the
    docstore would defeat the point of having a docstore. Real engines squeeze
    these to a byte each; these are full integers for now.

    The bloom filter is here for the same reason and pays off harder. Without
    it, a term absent from a segment still costs a dictionary read to discover
    that, and a query for a rare term pays that once per segment. With it, the
    answer is already in memory.
    """
    out = bytearray(struct.pack("<I", len(term_blocks)))
    for first_term, offset, length in term_blocks:
        encoded = first_term.encode("utf-8")
        out += struct.pack("<H", len(encoded))
        out += encoded
        out += struct.pack("<QI", offset, length)

    out += struct.pack("<I", len(doc_blocks))
    for first_doc_id, offset, length in doc_blocks:
        out += struct.pack("<IQI", first_doc_id, offset, length)

    out += struct.pack(f"<I{len(doc_lengths)}I", len(doc_lengths), *doc_lengths)
    out += bloom.to_bytes()
    return bytes(out)


def _decode_hotcache(
    data: bytes,
) -> tuple[
    list[tuple[str, int, int]], list[tuple[int, int, int]], list[int], BloomFilter
]:
    (term_block_count,) = struct.unpack_from("<I", data)
    pos = 4
    term_blocks = []
    for _ in range(term_block_count):
        (term_length,) = struct.unpack_from("<H", data, pos)
        pos += 2
        term = data[pos : pos + term_length].decode("utf-8")
        pos += term_length
        offset, length = struct.unpack_from("<QI", data, pos)
        pos += 12
        term_blocks.append((term, offset, length))

    (doc_block_count,) = struct.unpack_from("<I", data, pos)
    pos += 4
    doc_blocks = []
    for _ in range(doc_block_count):
        first_doc_id, offset, length = struct.unpack_from("<IQI", data, pos)
        pos += 16
        doc_blocks.append((first_doc_id, offset, length))

    (doc_count,) = struct.unpack_from("<I", data, pos)
    pos += 4
    doc_lengths = list(struct.unpack_from(f"<{doc_count}I", data, pos))
    pos += 4 * doc_count

    bloom, _ = BloomFilter.from_bytes(data, pos)
    return term_blocks, doc_blocks, doc_lengths, bloom


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def write_segment(index: SearchableIndex) -> bytes:
    """Serialize an index into segment bytes."""
    out = bytearray()

    # 1. Postings. Writing these first is what gives every term a byte range
    #    for the dictionary to point at.
    postings_offset = 0
    term_ranges: dict[str, tuple[int, int, int]] = {}
    for term in sorted(index.terms()):
        posting_list = index.postings(term)
        encoded = _encode_postings(posting_list.doc_ids, posting_list.freqs)
        term_ranges[term] = (posting_list.df, len(out), len(encoded))
        out += encoded
    postings_length = len(out) - postings_offset

    # 2. Docstore, in blocks. Only the few documents actually displayed are
    #    ever read, so this sits apart from the postings and is touched once
    #    per query at most.
    docstore_offset = len(out)
    doc_blocks: list[tuple[int, int, int]] = []
    for first in range(0, index.num_docs, DOCS_PER_BLOCK):
        docs = [
            index.document(i)
            for i in range(first, min(first + DOCS_PER_BLOCK, index.num_docs))
        ]
        # Deflated per block rather than per document. Compressing many small
        # JSON objects together lets the algorithm exploit the fact that every
        # one of them repeats the same field names, which a document at a time
        # cannot do. Block size therefore trades compression against read
        # amplification: a bigger block packs better but drags more unwanted
        # documents along when one of them is displayed.
        encoded = zlib.compress(
            json.dumps(docs, separators=(",", ":")).encode("utf-8"), 6
        )
        doc_blocks.append((first, len(out), len(encoded)))
        out += encoded
    docstore_length = len(out) - docstore_offset

    # 3. Term dictionary, blocked so one term costs one request.
    termdict_offset = len(out)
    term_blocks: list[tuple[str, int, int]] = []
    sorted_terms = sorted(term_ranges)
    for first in range(0, len(sorted_terms), TERMS_PER_BLOCK):
        chunk = sorted_terms[first : first + TERMS_PER_BLOCK]
        encoded = _encode_dict_block(
            [(term, *term_ranges[term]) for term in chunk]
        )
        term_blocks.append((chunk[0], len(out), len(encoded)))
        out += encoded
    termdict_length = len(out) - termdict_offset

    # 4. Hotcache, which needs the dictionary's offsets.
    hotcache_offset = len(out)
    doc_lengths = [index.doc_length(i) for i in range(index.num_docs)]
    out += _encode_hotcache(
        term_blocks, doc_blocks, doc_lengths, BloomFilter.for_terms(sorted_terms)
    )
    hotcache_length = len(out) - hotcache_offset

    # 5. Footer, which needs all of the above.
    timestamps = [
        doc["ts"]
        for i in range(index.num_docs)
        if isinstance((doc := index.document(i)).get("ts"), int)
    ]
    footer = Footer(
        version=SEGMENT_VERSION,
        postings_offset=postings_offset,
        postings_length=postings_length,
        docstore_offset=docstore_offset,
        docstore_length=docstore_length,
        termdict_offset=termdict_offset,
        termdict_length=termdict_length,
        hotcache_offset=hotcache_offset,
        hotcache_length=hotcache_length,
        num_docs=index.num_docs,
        num_terms=index.num_terms,
        num_postings=index.num_postings,
        total_length=sum(doc_lengths),
        min_ts=min(timestamps, default=0),
        max_ts=max(timestamps, default=0),
    )
    out += footer.pack()
    return bytes(out)


def write_segment_file(index: SearchableIndex, path: Path) -> int:
    """Write a segment to disk. Returns bytes written."""
    data = write_segment(index)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


class SegmentReader(SearchableIndex):
    """Search one segment without ever reading it whole.

    Opening costs two requests: the footer, fetched from the tail without
    knowing the file's size, and the hotcache it points to. Both are cached
    for the life of the reader, and that is safe with no invalidation logic
    whatsoever because a segment can never change. Immutability does not just
    make caching correct here, it deletes cache invalidation from the problem
    entirely.

    After that, a term costs at most one request for its dictionary block plus
    one for its postings, and a displayed document costs one for its docstore
    block. Dictionary and docstore blocks are cached too, since they are small
    and bounded; posting lists are not, because they are neither.
    """

    def __init__(self, store: ObjectStore, key: str) -> None:
        self._store = store
        self._key = key

        # Request 1: the last FOOTER_SIZE bytes. No prior knowledge needed,
        # not even the file's length.
        self.footer = Footer.unpack(store.get_suffix(key, FOOTER_SIZE))

        # Request 2: the hotcache, located by the footer.
        (
            self._term_blocks,
            self._doc_blocks,
            self._doc_lengths,
            self._bloom,
        ) = _decode_hotcache(
            store.get_range(key, self.footer.hotcache_offset, self.footer.hotcache_length)
        )
        self._first_terms = [block[0] for block in self._term_blocks]
        self._first_docs = [block[0] for block in self._doc_blocks]

        self._dict_cache: dict[int, dict[str, tuple[int, int, int]]] = {}
        self._doc_cache: dict[int, list[dict]] = {}
        # How many lookups the bloom filter answered outright. Each one is a
        # dictionary read that did not happen, which is the only way to see
        # what the filter is worth: it shows up as requests absent from the
        # counter rather than as anything present in it.
        self.bloom_rejections = 0

    @classmethod
    def open(cls, path: Path | str) -> SegmentReader:
        """Convenience for a segment sitting on the local filesystem."""
        path = Path(path)
        return cls(LocalStore(path.parent), path.name)

    # -- locating ----------------------------------------------------------

    def _dict_block(self, block_index: int) -> dict[str, tuple[int, int, int]]:
        cached = self._dict_cache.get(block_index)
        if cached is None:
            _, offset, length = self._term_blocks[block_index]
            cached = self._dict_cache[block_index] = _decode_dict_block(
                self._store.get_range(self._key, offset, length)
            )
        return cached

    def _entry(self, term: str) -> tuple[int, int, int] | None:
        """Find a term's (df, postings offset, postings length).

        Three levels, all but the last resolved in memory. The bloom filter
        rules the term out entirely, or the sparse index in the hotcache names
        the one dictionary block that could hold it. Binary searching the
        dictionary over the network instead would cost a round trip per probe,
        which is the naive design this format exists to avoid.
        """
        # The bloom filter answers from memory. A "definitely not" ends the
        # lookup here, which is the difference between one request per segment
        # and none at all when a term is absent, and rare terms are the common
        # case in real query logs.
        if term not in self._bloom:
            self.bloom_rejections += 1
            return None

        block_index = bisect_right(self._first_terms, term) - 1
        if block_index < 0:
            return None
        return self._dict_block(block_index).get(term)

    def _docstore_block(self, block_index: int) -> list[dict]:
        cached = self._doc_cache.get(block_index)
        if cached is None:
            _, offset, length = self._doc_blocks[block_index]
            cached = self._doc_cache[block_index] = json.loads(
                zlib.decompress(self._store.get_range(self._key, offset, length))
            )
        return cached

    # -- reading -----------------------------------------------------------

    def postings(self, term: str) -> PostingList | None:
        entry = self._entry(term)
        if entry is None:
            return None
        _, offset, length = entry
        return _decode_postings(self._store.get_range(self._key, offset, length))

    def contains(self, term: str) -> bool:
        """Presence without touching the postings.

        A bloom miss answers from memory for no requests at all; otherwise it
        costs the one dictionary block that could hold the term, which
        scoring would have had to read anyway.
        """
        return self._entry(term) is not None

    def df(self, term: str) -> int:
        """Document frequency without fetching the posting list.

        This is why df is stored in the dictionary: BM25 needs it for every
        query term, and reading a whole posting list to count its entries
        would be a wasted request every time.
        """
        entry = self._entry(term)
        return entry[0] if entry else 0

    def document(self, doc_id: int) -> dict:
        if not 0 <= doc_id < self.footer.num_docs:
            raise IndexError(f"doc id {doc_id} out of range")
        block_index = bisect_right(self._first_docs, doc_id) - 1
        block = self._docstore_block(block_index)
        return block[doc_id - self._first_docs[block_index]]

    def doc_length(self, doc_id: int) -> int:
        return self._doc_lengths[doc_id]

    def terms(self) -> Iterator[str]:
        """Every term, which means reading every dictionary block.

        Fine for introspection and tests, wrong for a query path. Nothing in
        the query path calls it.
        """
        for block_index in range(len(self._term_blocks)):
            yield from self._dict_block(block_index)

    def term_dfs(self) -> Iterator[tuple[str, int]]:
        """Terms with their document frequencies, reading only the dictionary."""
        for block_index in range(len(self._term_blocks)):
            for term, (df, _, _) in self._dict_block(block_index).items():
                yield term, df

    def most_common_terms(self, n: int = 10) -> list[tuple[str, int]]:
        # Overridden so it never touches the postings: the base class would
        # call df() term by term, and df is already in the dictionary.
        return sorted(self.term_dfs(), key=lambda pair: (-pair[1], pair[0]))[:n]

    # -- shape -------------------------------------------------------------

    @property
    def num_docs(self) -> int:
        return self.footer.num_docs

    @property
    def num_terms(self) -> int:
        return self.footer.num_terms

    @property
    def num_postings(self) -> int:
        return self.footer.num_postings

    @property
    def avg_doc_length(self) -> float:
        if not self.footer.num_docs:
            return 0.0
        return self.footer.total_length / self.footer.num_docs
