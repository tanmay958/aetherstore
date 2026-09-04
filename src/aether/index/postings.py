"""Posting lists and the set operations over them.

A posting list is every document containing one term. Almost all of an index's
bytes are these lists, so their shape drives the whole design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class PostingList:
    """Every document containing one term, in ascending document order.

    `doc_ids[i]` and `freqs[i]` describe the same document: the id, and how
    many times the term occurred in it.

    Parallel arrays rather than a list of pairs, because that is exactly the
    on-disk layout later: one stream of delta-encoded ids, one stream of
    frequencies. Matching the eventual layout now keeps the segment writer a
    translation rather than a redesign.

    Frequencies go unused until BM25 arrives, but they are recorded during
    indexing because recomputing them later would mean a second pass over
    every document.
    """

    doc_ids: list[int] = field(default_factory=list)
    freqs: list[int] = field(default_factory=list)

    def append(self, doc_id: int, freq: int) -> None:
        # Document ids are handed out in increasing order as events arrive, so
        # appends are always ascending and the list sorts itself for free.
        # Delta encoding and skip pointers both rest on that invariant, and
        # both fail silently if it breaks, so it is enforced rather than
        # assumed.
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

    Both inputs must be ascending and duplicate-free.

    A merge walk rather than `set(a) & set(b)`. Sets would be shorter and
    faster here, but the segment reader cannot use them: it decodes postings
    in sorted order out of a byte stream and never holds a whole list in
    memory. The walk is also what makes skip pointers possible, since skipping
    ahead only works on sorted input.
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
