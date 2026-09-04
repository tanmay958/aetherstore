"""Tests for the analyzer.

What the analyzer emits is the upper bound on what the index can ever find, so
these tests pin the tokenization rules rather than treating them as incidental.
"""

import pytest

from aether.index.analyzer import MIN_TERM_LENGTH, STOPWORDS, analyze_document, tokenize


def test_lowercases_and_splits_on_whitespace():
    assert tokenize("Samsung White Lite Smartphone") == [
        "samsung",
        "white",
        "lite",
        "smartphone",
    ]


def test_splits_a_dotted_category_path():
    """REES46 ships categories as "electronics.smartphone", so a search for
    "smartphone" only works because the dot separates tokens."""
    assert tokenize("appliances.kitchen.refrigerators") == [
        "appliances",
        "kitchen",
        "refrigerators",
    ]


def test_keeps_model_codes():
    """Derived model codes are the rare, high-cardinality terms that give BM25
    something genuinely discriminating to score."""
    assert "l486" in tokenize("Samsung White Lite Smartphone L486")


def test_drops_stopwords():
    assert tokenize("the best of the best") == ["best", "best"]
    assert all(word not in tokenize(" ".join(STOPWORDS)) for word in STOPWORDS)


def test_drops_single_characters():
    assert tokenize("5 x 3 cm") == ["cm"]
    assert all(len(term) >= MIN_TERM_LENGTH for term in tokenize("a b cd efg"))


def test_preserves_duplicates_and_order():
    """Term frequency is counted from this list, so collapsing duplicates here
    would silently destroy it. Order will matter later for phrase queries."""
    assert tokenize("red shoes red") == ["red", "shoes", "red"]


@pytest.mark.parametrize("value", ["", None])
def test_empty_input_yields_no_terms(value):
    assert tokenize(value) == []


def test_punctuation_and_symbols_separate_tokens():
    assert tokenize("wi-fi router, 5GHz!") == ["wi", "fi", "router", "5ghz"]


# --------------------------------------------------------------------------
# whole documents
# --------------------------------------------------------------------------


def test_pools_terms_from_every_text_field():
    doc = {
        "title": "Samsung White Lite Smartphone L486",
        "category": "electronics.smartphone",
        "brand": "samsung",
        "event_type": "purchase",
        "query": None,
    }
    terms = analyze_document(doc)
    assert "samsung" in terms
    assert "electronics" in terms
    assert "purchase" in terms


def test_a_term_in_two_fields_is_counted_twice():
    """Brand appears in both `brand` and the derived title, exactly as real
    product text repeats itself. That is a term frequency of 2, not 1."""
    doc = {"title": "Samsung Smartphone", "brand": "samsung"}
    assert analyze_document(doc).count("samsung") == 2


def test_missing_fields_contribute_nothing():
    """REES46 has no device and no queries, so those are permanently None."""
    assert analyze_document({"title": None, "brand": None, "query": None}) == []


def test_non_text_fields_are_not_indexed():
    """Prices and identifiers are for filtering and display, not full text."""
    terms = analyze_document({"title": "Sofa", "price": 415.71, "user_id": "555447699"})
    assert terms == ["sofa"]
