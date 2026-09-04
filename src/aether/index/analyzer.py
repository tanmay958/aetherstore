"""Turning text into index terms.

The analyzer is the first thing to touch a document, and it decides what counts
as a word. That makes it more consequential than it looks: the index can only
ever answer questions about terms the analyzer chose to emit. A token dropped
here cannot be recovered by any amount of cleverness downstream.

The rules are deliberately plain. Production engines add stemming, synonyms,
and language detection on top, but those are refinements of this shape rather
than a different shape.

One thing worth noticing on REES46 data: `category` arrives as a dotted path
like "electronics.smartphone", and the token pattern treats the dot as a
separator, so it indexes as two terms. That is why a search for "smartphone"
works even though no field literally contains that word on its own.
"""

from __future__ import annotations

import re

from aether.events import TEXT_FIELDS

# Applied after lowercasing, so the character class only needs lowercase.
# Everything else, including dots, hyphens, and whitespace, separates tokens.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Words common enough to appear in nearly every document. Indexing them costs
# space and buys nothing: a term present in most documents cannot discriminate
# between them, which is the same intuition BM25 formalizes later.
STOPWORDS = frozenset(
    "a an and are as at be by for from has in is it of on or that the to was with".split()
)

# Single characters are almost always fragments of something else, such as the
# "x" in "5 x 3". Model codes like "L486" survive, since they are four long.
MIN_TERM_LENGTH = 2


def tokenize(text: str) -> list[str]:
    """Break a string into index terms.

    Duplicates and order are both preserved, because the caller needs to count
    how often each term occurs to record term frequency, and positions will
    matter later for phrase queries.

        >>> tokenize("Samsung White Lite Smartphone L486")
        ['samsung', 'white', 'lite', 'smartphone', 'l486']
        >>> tokenize("electronics.smartphone")
        ['electronics', 'smartphone']
    """
    if not text:
        return []
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) >= MIN_TERM_LENGTH and token not in STOPWORDS
    ]


def analyze_document(doc: dict) -> list[str]:
    """Extract every index term from one event.

    Terms from all text fields are pooled into a single flat list, so a search
    for "samsung" matches whether it appeared in the brand or in the title.
    Keeping fields separate would allow field-scoped queries like
    `brand:samsung`, which is a later refinement.

    Fields the source did not provide are None and contribute nothing.
    """
    terms: list[str] = []
    for field in TEXT_FIELDS:
        value = doc.get(field)
        if value:
            terms.extend(tokenize(str(value)))
    return terms
