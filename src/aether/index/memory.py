"""The in-memory inverted index.

The simplest thing that is unmistakably correct: a dict from term to a sorted
list of document ids. It is not fast and it does not scale, and neither
matters, because its job is to be *trustworthy*.

Every later stage of the storage engine, the segment file, the bit-packed
postings codec, the byte-range reader, is verified by running the same queries
against this and asserting identical results. When a compressed reader and
this dict disagree, this dict is right. That is why it exists before any file
format does: a segment reader cannot be validated without something already
known to be correct.

Query logic lives in SearchableIndex, shared with every other backend.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from aether.index.analyzer import analyze_document
from aether.index.base import SearchableIndex
from aether.index.postings import PostingList


class MemoryIndex(SearchableIndex):
    """An inverted index held entirely in RAM.

    Document ids are local to this index and assigned sequentially from zero,
    exactly as they will be inside a single segment. They double as the
    position of the document in `_documents`, so retrieving a hit costs an
    array index rather than a lookup table.
    """

    def __init__(self) -> None:
        self._postings: dict[str, PostingList] = {}
        self._documents: list[dict] = []
        self._doc_lengths: list[int] = []
        self._total_length = 0

    # -- building ----------------------------------------------------------

    def add(self, doc: dict) -> int:
        """Index one document and return the id assigned to it."""
        doc_id = len(self._documents)
        self._documents.append(doc)

        terms = analyze_document(doc)
        self._doc_lengths.append(len(terms))
        self._total_length += len(terms)

        for term, freq in Counter(terms).items():
            posting_list = self._postings.get(term)
            if posting_list is None:
                posting_list = self._postings[term] = PostingList()
            posting_list.append(doc_id, freq)

        return doc_id

    def add_all(self, docs: Iterable[dict]) -> int:
        """Index many documents, returning how many were added."""
        count = 0
        for doc in docs:
            self.add(doc)
            count += 1
        return count

    # -- reading -----------------------------------------------------------

    def postings(self, term: str) -> PostingList | None:
        return self._postings.get(term)

    def document(self, doc_id: int) -> dict:
        return self._documents[doc_id]

    def doc_length(self, doc_id: int) -> int:
        return self._doc_lengths[doc_id]

    def terms(self) -> Iterable[str]:
        return self._postings.keys()

    @property
    def num_docs(self) -> int:
        return len(self._documents)

    @property
    def num_terms(self) -> int:
        return len(self._postings)

    @property
    def num_postings(self) -> int:
        return sum(len(pl) for pl in self._postings.values())

    @property
    def avg_doc_length(self) -> float:
        return self._total_length / self.num_docs if self._documents else 0.0


def build_index(docs: Iterable[dict]) -> MemoryIndex:
    """Convenience constructor: index an iterable of events."""
    index = MemoryIndex()
    index.add_all(docs)
    return index
