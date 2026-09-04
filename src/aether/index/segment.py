"""Segments: an index serialized into one self-contained file.

A segment is a file. Not a concept, not a data structure, a file sitting in
storage. What makes it useful is that it is *self-contained*: it holds its own
term dictionary, its own posting lists, and its own copy of the documents, so
a segment can be searched with nothing else present. That is what lets any
machine search any segment independently, with no coordination.

Two facts force segments to exist. RAM runs out, so an index cannot simply
stay in memory. And object storage offers no way to modify an object, only to
replace it whole, so the obvious design of one big index file that keeps
growing is not merely slow but unavailable. The only move is to write new
files, and never touch them again.

This step establishes the contract and the round-trip, not the encoding. The
payload here is JSON, which is honest about being temporary: it is roughly the
size of the raw input and the reader loads all of it into memory, which is
exactly what a real segment must not do. Step 3 replaces the body with the
five-section binary layout and reads it through byte ranges alone. Because
SegmentReader satisfies the same interface as MemoryIndex, the tests written
now carry over unchanged and prove the swap did not break anything.

The header is already real, though:

    b"ATHR"  magic, so a wrong file fails loudly instead of weirdly
    u32      format version, so old segments stay readable
    ...      payload

    python -m aether.index.build tests/fixtures/rees46_sample.csv out.seg
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterable

from aether.index.base import SearchableIndex
from aether.index.postings import PostingList

SEGMENT_MAGIC = b"ATHR"
SEGMENT_VERSION = 1

# 4 byte magic + 4 byte little-endian version.
_HEADER = struct.Struct("<4sI")
HEADER_SIZE = _HEADER.size


def write_segment(index: SearchableIndex) -> bytes:
    """Serialize an index into segment bytes."""
    payload = {
        # Counts are stored rather than derived, because a reader must be able
        # to report the shape of a segment without walking its postings. This
        # is what the fixed-size footer will carry in step 3.
        "num_docs": index.num_docs,
        "num_terms": index.num_terms,
        "num_postings": index.num_postings,
        "doc_lengths": [index.doc_length(i) for i in range(index.num_docs)],
        "documents": [index.document(i) for i in range(index.num_docs)],
        # Parallel arrays, matching PostingList and the eventual on-disk
        # layout: one stream of ids, one of frequencies.
        "terms": {
            term: [index.postings(term).doc_ids, index.postings(term).freqs]
            for term in sorted(index.terms())
        },
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return _HEADER.pack(SEGMENT_MAGIC, SEGMENT_VERSION) + body


def write_segment_file(index: SearchableIndex, path: Path) -> int:
    """Write a segment to disk. Returns bytes written."""
    data = write_segment(index)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


class SegmentReader(SearchableIndex):
    """Read-only search over one segment.

    Loads the whole payload up front, which is the thing the format exists to
    avoid and which step 3 fixes. The interface it presents is the one that
    survives: identical to MemoryIndex, so callers and tests never learn where
    the postings actually came from.
    """

    def __init__(self, payload: dict) -> None:
        self._num_docs: int = payload["num_docs"]
        self._num_terms: int = payload["num_terms"]
        self._num_postings: int = payload["num_postings"]
        self._doc_lengths: list[int] = payload["doc_lengths"]
        self._documents: list[dict] = payload["documents"]
        self._terms: dict[str, list[list[int]]] = payload["terms"]

    # -- opening -----------------------------------------------------------

    @classmethod
    def from_bytes(cls, data: bytes) -> SegmentReader:
        if len(data) < HEADER_SIZE:
            raise ValueError(
                f"not a segment: {len(data)} bytes is shorter than the {HEADER_SIZE} "
                "byte header"
            )
        magic, version = _HEADER.unpack_from(data)
        if magic != SEGMENT_MAGIC:
            raise ValueError(
                f"not a segment: magic was {magic!r}, expected {SEGMENT_MAGIC!r}"
            )
        if version != SEGMENT_VERSION:
            raise ValueError(
                f"segment format version {version} is not supported "
                f"(this build reads version {SEGMENT_VERSION})"
            )
        try:
            payload = json.loads(data[HEADER_SIZE:].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"segment body is corrupt: {exc}") from exc
        return cls(payload)

    @classmethod
    def open(cls, path: Path) -> SegmentReader:
        return cls.from_bytes(Path(path).read_bytes())

    # -- reading -----------------------------------------------------------

    def postings(self, term: str) -> PostingList | None:
        entry = self._terms.get(term)
        if entry is None:
            return None
        doc_ids, freqs = entry
        return PostingList(doc_ids=doc_ids, freqs=freqs)

    def document(self, doc_id: int) -> dict:
        return self._documents[doc_id]

    def doc_length(self, doc_id: int) -> int:
        return self._doc_lengths[doc_id]

    def terms(self) -> Iterable[str]:
        return self._terms.keys()

    @property
    def num_docs(self) -> int:
        return self._num_docs

    @property
    def num_terms(self) -> int:
        return self._num_terms

    @property
    def num_postings(self) -> int:
        return self._num_postings

    @property
    def avg_doc_length(self) -> float:
        if not self._num_docs:
            return 0.0
        return sum(self._doc_lengths) / self._num_docs
