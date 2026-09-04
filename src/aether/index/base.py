"""The read interface every index implementation satisfies.

There are two implementations, and later a third: the in-memory dict, a
segment file read from disk, and eventually a segment read over byte ranges
from object storage. Query logic is identical for all of them, because a
conjunction is just a walk over posting lists no matter where those postings
came from.

Putting that logic here rather than in each implementation buys the thing that
matters most: the oracle test can be parameterized over every implementation
at once, so a new backend is proven against the brute-force scan the moment it
exists.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from aether.index.analyzer import tokenize
from aether.index.postings import PostingList, intersect, union


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
        """Term count for one document, used by BM25 to stop long documents
        from scoring highly purely by containing more words."""

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

        The number that actually determines segment size: it is how many
        document ids have to be written down, and almost all of an index's
        bytes are document ids.
        """

    @property
    @abstractmethod
    def avg_doc_length(self) -> float: ...

    # -- shared query logic ------------------------------------------------

    def df(self, term: str) -> int:
        posting_list = self.postings(term)
        return posting_list.df if posting_list else 0

    def documents(self, doc_ids: Iterable[int]) -> list[dict]:
        return [self.document(doc_id) for doc_id in doc_ids]

    def search_and(self, query: str) -> list[int]:
        """Documents containing every term in the query.

        Posting lists are intersected shortest-first. An intersection can only
        shrink, so starting from the rarest term keeps every subsequent walk
        as short as possible: pairing a 3-document list with a 50,000-document
        one costs 50,003 steps, while two 50,000-document lists cost 100,000.
        """
        terms = set(tokenize(query))
        if not terms:
            return []

        lists: list[list[int]] = []
        for term in terms:
            posting_list = self.postings(term)
            if posting_list is None:
                # One absent term collapses the whole conjunction.
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
        result: list[int] = []
        for term in set(tokenize(query)):
            posting_list = self.postings(term)
            if posting_list is not None:
                result = union(result, posting_list.doc_ids)
        return result

    def most_common_terms(self, n: int = 10) -> list[tuple[str, int]]:
        return sorted(
            ((term, self.df(term)) for term in self.terms()),
            key=lambda pair: (-pair[1], pair[0]),
        )[:n]
