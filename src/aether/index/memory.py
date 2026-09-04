"""The in-memory inverted index.

This is the simplest thing that is unmistakably correct: a dict from term to a
sorted list of document ids. It is not fast and it does not scale, and neither
matters, because its job is to be *trustworthy*.

Every later stage of the storage engine, the binary segment format, the
bit-packed postings codec, the byte-range reader, gets verified by running the
same queries against this file and asserting identical results. When the
compressed reader and this dict disagree, this dict is right. That is why it
is built before any file format exists: you cannot validate a segment reader
without something already known to be correct.

Two decisions here look like over-engineering and are not.

Posting lists are kept as parallel arrays of doc ids and frequencies, rather
than a list of pairs, because that is exactly how they are laid out on disk
later: one stream of delta-encoded ids, one stream of frequencies. Matching
the eventual layout now keeps the segment writer a translation rather than a
redesign.

Intersection is a two-pointer merge walk rather than `set(a) & set(b)`. Sets
would be shorter and faster *here*, but the segment reader cannot use them: it
decodes postings in sorted order out of a byte stream and never holds a whole
list in memory. The merge walk is also what makes skip pointers possible,
since skipping ahead only works on a sorted walk.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from aether.index.analyzer import analyze_document, tokenize


@dataclass
class PostingList:
    """Every document containing one term, in ascending document order.

    `doc_ids[i]` and `freqs[i]` describe the same document: the id, and how
    many times the term occurred in it. Frequencies are unused until BM25
    arrives, but they are recorded now because recomputing them later would
    mean a second pass over every document.
    """

    doc_ids: list[int] = field(default_factory=list)
    freqs: list[int] = field(default_factory=list)

    def append(self, doc_id: int, freq: int) -> None:
        # Document ids are handed out in increasing order as events arrive, so
        # appends are always ascending and the list sorts itself for free.
        # That invariant is what delta encoding and skip pointers both rest
        # on, so it is enforced rather than assumed.
        if self.doc_ids and doc_id <= self.doc_ids[-1]:
            raise ValueError(
                f"doc ids must ascend: appending {doc_id} after {self.doc_ids[-1]}"
            )
        self.doc_ids.append(doc_id)
        self.freqs.append(freq)

    @property
    def df(self) -> int:
        """Document frequency: how many documents contain this term."""
        return len(self.doc_ids)

    def freq_in(self, doc_id: int) -> int:
        """Term frequency within one document, or 0 if absent."""
        try:
            return self.freqs[self.doc_ids.index(doc_id)]
        except ValueError:
            return 0

    def __len__(self) -> int:
        return len(self.doc_ids)


def intersect(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Documents present in both lists, in O(len(a) + len(b)).

    Both inputs must be ascending and duplicate-free. Walking two sorted lists
    with a pointer each is the operation an AND query is built from, and it is
    the same walk the segment reader performs over decoded postings.
    """
    out: list[int] = []
    i = j = 0
    len_a, len_b = len(a), len(b)
    while i < len_a and j < len_b:
        av, bv = a[i], b[j]
        if av == bv:
            out.append(av)
            i += 1
            j += 1
        elif av < bv:
            i += 1
        else:
            j += 1
    return out


def union(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Documents present in either list, still ascending and duplicate-free."""
    out: list[int] = []
    i = j = 0
    len_a, len_b = len(a), len(b)
    while i < len_a and j < len_b:
        av, bv = a[i], b[j]
        if av == bv:
            out.append(av)
            i += 1
            j += 1
        elif av < bv:
            out.append(av)
            i += 1
        else:
            out.append(bv)
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


class MemoryIndex:
    """An inverted index held entirely in RAM.

    Document ids are local to this index and assigned sequentially from zero,
    exactly as they will be inside a single segment later. They are also the
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

    def df(self, term: str) -> int:
        posting_list = self._postings.get(term)
        return posting_list.df if posting_list else 0

    def document(self, doc_id: int) -> dict:
        return self._documents[doc_id]

    def documents(self, doc_ids: Iterable[int]) -> list[dict]:
        return [self._documents[doc_id] for doc_id in doc_ids]

    def doc_length(self, doc_id: int) -> int:
        """Term count for one document. BM25 uses this to stop long documents
        from scoring highly purely by containing more words."""
        return self._doc_lengths[doc_id]

    # -- searching ---------------------------------------------------------

    def search_and(self, query: str) -> list[int]:
        """Documents containing every term in the query.

        Posting lists are intersected shortest-first. An intersection can only
        ever shrink, so starting from the rarest term keeps every subsequent
        walk as short as possible: pairing a 3-document list with a
        50,000-document one costs 50,003 steps, while two 50,000-document
        lists cost 100,000.
        """
        terms = set(tokenize(query))
        if not terms:
            return []

        lists: list[list[int]] = []
        for term in terms:
            posting_list = self._postings.get(term)
            if posting_list is None:
                # One absent term makes the whole conjunction empty.
                return []
            lists.append(posting_list.doc_ids)

        lists.sort(key=len)
        result = lists[0]
        for other in lists[1:]:
            result = intersect(result, other)
            if not result:
                break
        return list(result)

    def search_or(self, query: str) -> list[int]:
        """Documents containing at least one term in the query."""
        terms = set(tokenize(query))
        result: list[int] = []
        for term in terms:
            posting_list = self._postings.get(term)
            if posting_list is not None:
                result = union(result, posting_list.doc_ids)
        return result

    # -- introspection -----------------------------------------------------

    @property
    def num_docs(self) -> int:
        return len(self._documents)

    @property
    def num_terms(self) -> int:
        return len(self._postings)

    @property
    def num_postings(self) -> int:
        """Total entries across all posting lists.

        This is the number that actually determines segment size later: it is
        how many document ids have to be written down, and almost all of an
        index's bytes are document ids.
        """
        return sum(len(pl) for pl in self._postings.values())

    @property
    def avg_doc_length(self) -> float:
        return self._total_length / self.num_docs if self._documents else 0.0

    def most_common_terms(self, n: int = 10) -> list[tuple[str, int]]:
        return sorted(
            ((term, pl.df) for term, pl in self._postings.items()),
            key=lambda pair: (-pair[1], pair[0]),
        )[:n]


def build_index(docs: Iterable[dict]) -> MemoryIndex:
    """Convenience constructor: index an iterable of events."""
    index = MemoryIndex()
    index.add_all(docs)
    return index
