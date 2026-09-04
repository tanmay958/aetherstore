"""The read interface every index implementation satisfies.

There are two implementations, and later a third: the in-memory dict, a
segment file read by byte range, and eventually a segment read over the
network from object storage. Query logic is identical for all of them, because
a conjunction is a walk over posting lists no matter where those postings came
from.

Putting that logic here rather than in each implementation buys the thing that
matters most: the oracle test is parameterized over every implementation at
once, so a new backend is proven against the brute-force scan the moment it
exists.
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Literal

from aether.index.analyzer import tokenize
from aether.index.postings import PostingList, intersect, union
from aether.index.scorer import DEFAULT_SCORER, BM25

Mode = Literal["and", "or"]


@dataclass(frozen=True)
class CorpusStats:
    """Collection-wide statistics to score against, overriding a segment's own.

    BM25 asks two questions about the collection: how many documents are in
    it, and how many of them contain this term. A single segment can only
    answer for itself, and its answers are wrong for an index made of many
    segments. A term appearing in three documents of a small segment looks
    rare there and common in a large one, so the same document earns
    different scores depending on which segment it happened to land in.

    Supplying these makes every segment score against the same denominator,
    at the cost of an extra round trip to collect them first. See
    `Coordinator.search`, which offers both and explains the trade.
    """

    num_docs: int
    avg_doc_length: float
    dfs: dict[str, int]


@dataclass(frozen=True)
class Hit:
    """One ranked result."""

    doc_id: int
    score: float


@dataclass(frozen=True)
class SearchResult:
    """The best hits, plus how many matched in total.

    The total is returned rather than left to the caller because recomputing
    it means matching a second time, and on a segment that means fetching
    every posting list again. A displayed "top 10 of 1,832" should not cost
    twice as many requests as "top 10".
    """

    hits: list[Hit]
    total: int

    def __len__(self) -> int:
        return len(self.hits)

    def __iter__(self):
        return iter(self.hits)


class SearchableIndex(ABC):
    """Anything that can answer a query over an inverted index."""

    # -- what each implementation must provide -----------------------------

    @abstractmethod
    def postings(self, term: str) -> PostingList | None:
        """The posting list for a term, or None if the term is absent."""

    @abstractmethod
    def document(self, doc_id: int) -> dict:
        """The original event behind a document id."""

    @abstractmethod
    def doc_length(self, doc_id: int) -> int:
        """Term count for one document, which BM25 divides by so that long
        documents do not score highly purely by containing more words."""

    @abstractmethod
    def terms(self) -> Iterable[str]:
        """Every indexed term."""

    @property
    @abstractmethod
    def num_docs(self) -> int: ...

    @property
    @abstractmethod
    def num_terms(self) -> int: ...

    @property
    @abstractmethod
    def num_postings(self) -> int:
        """Total entries across all posting lists.

        The number that determines segment size: it is how many document ids
        have to be written down, and almost all of an index's bytes are
        document ids.
        """

    @property
    @abstractmethod
    def avg_doc_length(self) -> float: ...

    # -- fetching ----------------------------------------------------------

    def df(self, term: str) -> int:
        posting_list = self.postings(term)
        return posting_list.df if posting_list else 0

    def contains(self, term: str) -> bool:
        """Whether a term appears at all, without fetching its posting list.

        The default is honest but wasteful. A segment overrides it to answer
        from a bloom filter and a dictionary block, both far cheaper than the
        postings themselves.
        """
        return self.postings(term) is not None

    def documents(self, doc_ids: Iterable[int]) -> list[dict]:
        return [self.document(doc_id) for doc_id in doc_ids]

    def _postings_for(
        self, query: str, *, require_all: bool
    ) -> tuple[list[tuple[str, PostingList]], bool]:
        """Posting lists for the query's terms, fetched once each.

        Fetched once matters on a segment, where every call is a request.
        Under `require_all` a missing term returns immediately without
        fetching the rest, because one absent term collapses a conjunction and
        the remaining reads would buy nothing.
        """
        terms = sorted(set(tokenize(query)))

        # For a conjunction, establish that every term is present before
        # fetching any postings. Otherwise a query for one common and one
        # absent term pays to download the common term's posting list, then
        # discards it on discovering the conjunction is empty. Presence is
        # cheap to check and postings are not.
        if require_all and not all(self.contains(term) for term in terms):
            return [], False

        lists: list[tuple[str, PostingList]] = []
        for term in terms:
            posting_list = self.postings(term)
            if posting_list is None:
                if require_all:
                    return [], False
                continue
            # The term travels with its postings because global scoring needs
            # to look up a corpus-wide document frequency by name.
            lists.append((term, posting_list))
        return lists, True

    # -- matching ----------------------------------------------------------

    @staticmethod
    def _intersect_all(lists: list[tuple[str, PostingList]]) -> list[int]:
        """Shortest list first. An intersection can only shrink, so starting
        from the rarest term keeps every subsequent walk as short as possible:
        pairing a 3-document list with a 50,000-document one costs 50,003
        steps, while two 50,000-document lists cost 100,000."""
        ordered = sorted(lists, key=lambda pair: pair[1].df)
        result = ordered[0][1].doc_ids
        for _, posting_list in ordered[1:]:
            result = intersect(result, posting_list.doc_ids)
            if not result:
                break
        return list(result)

    @staticmethod
    def _union_all(lists: list[tuple[str, PostingList]]) -> list[int]:
        result: list[int] = []
        for _, posting_list in lists:
            result = union(result, posting_list.doc_ids)
        return result

    def search_and(self, query: str) -> list[int]:
        """Documents containing every term, in document order."""
        lists, complete = self._postings_for(query, require_all=True)
        if not complete or not lists:
            return []
        return self._intersect_all(lists)

    def search_or(self, query: str) -> list[int]:
        """Documents containing at least one term, in document order."""
        lists, _ = self._postings_for(query, require_all=False)
        return self._union_all(lists)

    # -- ranking -----------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        mode: Mode = "and",
        scorer: BM25 = DEFAULT_SCORER,
        corpus: CorpusStats | None = None,
    ) -> SearchResult:
        """The best `top_k` documents for a query, most relevant first.

        Costs no storage requests beyond the posting lists matching already
        needed: document frequency arrives with the dictionary block, term
        frequency with the postings, and document lengths with the hotcache.
        Every one of those was recorded in an earlier step for this moment.
        """
        if mode not in ("and", "or"):
            raise ValueError(f"mode must be 'and' or 'or', got {mode!r}")

        lists, complete = self._postings_for(query, require_all=mode == "and")
        if not lists or not complete:
            return SearchResult([], 0)

        candidates = (
            self._intersect_all(lists) if mode == "and" else self._union_all(lists)
        )
        if not candidates:
            return SearchResult([], 0)

        scores = self._accumulate(lists, candidates, scorer, corpus)
        # Negated so the heap sorts descending by score, and ties fall back to
        # ascending document id, which keeps results stable run to run.
        ranked = heapq.nsmallest(
            top_k, ((-score, doc_id) for doc_id, score in scores.items())
        )
        return SearchResult(
            [Hit(doc_id, -negated) for negated, doc_id in ranked], len(candidates)
        )

    def _accumulate(
        self,
        lists: list[tuple[str, PostingList]],
        candidates: list[int],
        scorer: BM25,
        corpus: CorpusStats | None = None,
    ) -> dict[int, float]:
        """Sum each term's contribution across the candidate documents.

        Both the posting list and the candidates are ascending, so the term
        frequency for each candidate is found by the same merge walk the
        intersection uses, rather than by building a lookup table.
        """
        num_docs = corpus.num_docs if corpus else self.num_docs
        avg_length = corpus.avg_doc_length if corpus else self.avg_doc_length
        scores = dict.fromkeys(candidates, 0.0)

        for term, posting_list in lists:
            # Corpus-wide document frequency when supplied, so segments score
            # against the same denominator; otherwise this segment's own.
            df = corpus.dfs.get(term, posting_list.df) if corpus else posting_list.df
            # No short-circuit on a small idf: a term present in every
            # document scores near zero but not at zero, and dropping it would
            # change results rather than only saving work.
            idf = scorer.idf(num_docs, df)
            doc_ids, freqs = posting_list.doc_ids, posting_list.freqs
            i = j = 0
            len_postings, len_candidates = len(doc_ids), len(candidates)
            while i < len_postings and j < len_candidates:
                posting_doc, candidate = doc_ids[i], candidates[j]
                if posting_doc == candidate:
                    scores[candidate] += scorer.term_score(
                        freqs[i], self.doc_length(candidate), avg_length, idf
                    )
                    i += 1
                    j += 1
                elif posting_doc < candidate:
                    i += 1
                else:
                    j += 1
        return scores

    # -- introspection -----------------------------------------------------

    def most_common_terms(self, n: int = 10) -> list[tuple[str, int]]:
        return sorted(
            ((term, self.df(term)) for term in self.terms()),
            key=lambda pair: (-pair[1], pair[0]),
        )[:n]
