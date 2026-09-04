"""BM25 relevance scoring.

Until now results came back in document id order, which is arbitrary. Ranking
means answering "which of these matches is most about the query", and BM25 is
the standard answer: it is what Lucene, Elasticsearch, and every production
text search engine use by default, and it has held that position for roughly
thirty years.

    score(D, Q) = sum over query terms t of

                                  tf * (k1 + 1)
        idf(t)  x  ------------------------------------------
                   tf + k1 * (1 - b + b * |D| / avgdl)

Three ideas are packed into that, and each one corrects a way the obvious
approach gets ranking wrong.

**Rare terms matter more.** A document matching "refrigerators" tells you far
more than one matching "view", which appears in most of the corpus. That is
the idf factor, and it is why document frequency was stored in the term
dictionary in step 3.

**Term frequency saturates.** A document mentioning "samsung" fifty times is
not fifty times more relevant than one mentioning it once; it is a bit more
relevant, and then it stops mattering. The tf term is a hyperbola that rises
steeply and flattens, with k1 setting where the knee falls. Naive tf-idf
misses this and rewards keyword stuffing.

**Long documents are penalized.** A long document contains more words, so it
matches more queries by accident. Dividing by length relative to the average
corrects for that, with b controlling how aggressively. This is why document
lengths went into the hotcache in step 3 rather than the docstore: BM25 needs
one per candidate, and a query can produce thousands of candidates, so
fetching them from the docstore would defeat having a docstore at all.

Every input BM25 needs was already being recorded before it existed: df in the
dictionary, tf in the postings, document length in the hotcache, and the
average in the footer. Scoring therefore costs no storage requests beyond the
posting lists a query had to read anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Defaults from the literature, and the same ones Lucene ships.
DEFAULT_K1 = 1.2
DEFAULT_B = 0.75


@dataclass(frozen=True)
class BM25:
    """The scoring function, with its two knobs.

    k1 controls how fast term frequency saturates. At 0 a term either occurs
    or does not and repetition is worth nothing; as it grows, extra
    occurrences keep counting for longer. 1.2 puts the knee around three or
    four occurrences.

    b controls length normalization. At 0 length is ignored entirely, at 1 the
    score is fully divided by relative length. 0.75 is the usual compromise.
    """

    k1: float = DEFAULT_K1
    b: float = DEFAULT_B

    def __post_init__(self) -> None:
        if self.k1 < 0:
            raise ValueError(f"k1 must be non-negative, got {self.k1}")
        if not 0.0 <= self.b <= 1.0:
            raise ValueError(f"b must be between 0 and 1, got {self.b}")

    def idf(self, num_docs: int, df: int) -> float:
        """How much a term's presence tells you, given how common it is.

        The textbook form, ln((N - df + 0.5) / (df + 0.5)), goes negative once
        a term appears in more than about half the corpus, which lets a common
        term actively subtract from a document's score and can rank a document
        below one that matched fewer terms. The +1 inside the logarithm is
        Lucene's fix: it keeps the value positive everywhere while preserving
        the ordering, so a ubiquitous term contributes almost nothing rather
        than something harmful.
        """
        if df <= 0:
            return 0.0
        return math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))

    def term_score(
        self, tf: int, doc_length: int, avg_doc_length: float, idf: float
    ) -> float:
        """One term's contribution to one document's score."""
        if tf <= 0:
            return 0.0
        # An empty index has no meaningful average; fall back to no length
        # normalization rather than dividing by zero.
        norm = doc_length / avg_doc_length if avg_doc_length > 0 else 1.0
        denominator = tf + self.k1 * (1.0 - self.b + self.b * norm)
        return idf * (tf * (self.k1 + 1.0)) / denominator


DEFAULT_SCORER = BM25()
