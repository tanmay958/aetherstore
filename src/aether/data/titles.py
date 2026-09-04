"""Deterministic product titles derived from REES46 product ids.

REES46 has no product name column. Its only text is `brand` (frequently empty)
and `category_code`, a dotted taxonomy path that is also frequently empty. That
yields roughly three index terms per document. Real product catalogs carry five
to ten, and the difference changes the shape of the index being measured: a
smaller term dictionary, longer and less varied posting lists, and almost
nothing for BM25's document-length normalization to work with.

So a title is derived per product. Two rules keep it honest.

1. Only the adjectives are invented. The brand and the category leaf come
   straight out of the file. A product with neither gets no title at all,
   rather than one conjured out of nothing.

2. A title is a pure function of the product id, so it is stable forever. Load
   a hundred thousand rows today and ten million tomorrow, and product 1005105
   is titled identically both times. That rules out a counter or a shared RNG
   stream, and it specifically rules out the builtin `hash()`, which is salted
   per process and would hand out different titles on every run. CRC32 is used
   instead: stable across processes, machines, and Python versions.

This is sound for storage engine measurements, which depend on the statistical
shape of the text rather than on whether it is true: vocabulary size, term
skew, document length. It is not sound for relevance claims, but REES46 ships
no relevance judgments, so those were never on the table.

    >>> derive_title("1005105", "samsung", "electronics.smartphone")
    'Samsung Silver Compact Smartphone K281'
"""

from __future__ import annotations

import bisect
import zlib
from functools import lru_cache

# Weights make the derived vocabulary skewed rather than uniform, because BM25
# scores a rare term above a common one and a flat distribution would leave it
# nothing to discriminate on. "black" lands on roughly twenty times as many
# products as "copper".
_COLORS = [
    ("black", 22), ("white", 18), ("silver", 14), ("grey", 11), ("blue", 9),
    ("red", 7), ("gold", 5), ("green", 4), ("beige", 3), ("rose", 2),
    ("copper", 1),
]

_MODIFIERS = [
    ("classic", 20), ("compact", 16), ("premium", 13), ("portable", 10),
    ("wireless", 8), ("essential", 7), ("lite", 5), ("pro", 4),
    ("max", 3), ("eco", 2), ("ultra", 1),
]


def _table(pairs: list[tuple[str, int]]) -> tuple[list[str], list[int], int]:
    """Precompute cumulative weights so a pick is one binary search."""
    values: list[str] = []
    cumulative: list[int] = []
    total = 0
    for value, weight in pairs:
        total += weight
        values.append(value)
        cumulative.append(total)
    return values, cumulative, total


_COLOR_TABLE = _table(_COLORS)
_MODIFIER_TABLE = _table(_MODIFIERS)


def _pick(table: tuple[list[str], list[int], int], digest: int) -> str:
    """Weighted choice driven by a hash rather than an RNG, so it is stable."""
    values, cumulative, total = table
    return values[bisect.bisect_right(cumulative, digest % total)]


def _model_code(digest: int) -> str:
    """A letter and three digits, e.g. "K281".

    Real catalogs carry model codes, and they matter here for a reason beyond
    realism: they are high-cardinality, so nearly every one has a document
    frequency of 1. That long tail of rare terms is what exercises the term
    dictionary's front coding and gives BM25 something genuinely rare to score.
    """
    return f"{chr(65 + digest % 26)}{digest // 26 % 1000:03d}"


# Bounded on purpose. A month of REES46 holds roughly 170k distinct products,
# so this covers a full month without growing without limit over seven. A miss
# costs three CRC32 calls, which is cheap enough that eviction does not matter.
@lru_cache(maxsize=200_000)
def derive_title(
    product_id: str, brand: str | None, category_code: str | None
) -> str | None:
    """Build a stable title for a product, or None if there is nothing real to
    anchor it to.

    The returned string interleaves real and derived words:

        Samsung   Silver    Compact    Smartphone   K281
        ^real     ^derived  ^derived   ^real        ^derived
    """
    leaf = None
    if category_code:
        leaf = category_code.rsplit(".", 1)[-1].replace("_", " ").strip()

    # Neither a brand nor a category means no real anchor exists. Inventing a
    # title from pure hash output would be fabrication, so decline instead.
    if not brand and not leaf:
        return None

    # Separate CRC32 seeds per slot. Shifting one digest would reuse correlated
    # bits and visibly couple color to modifier.
    base = zlib.crc32(product_id.encode("utf-8"))

    parts: list[str] = []
    if brand:
        parts.append(brand.strip().title())
    parts.append(_pick(_COLOR_TABLE, zlib.crc32(b"color", base)).title())
    parts.append(_pick(_MODIFIER_TABLE, zlib.crc32(b"modifier", base)).title())
    if leaf:
        parts.append(leaf.title())
    parts.append(_model_code(zlib.crc32(b"model", base)))
    return " ".join(parts)
