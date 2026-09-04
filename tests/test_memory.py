"""Tests for building the in-memory index.

Query behaviour lives in test_index_contract.py, which runs against every
backend. What is specific to this class is how the index gets built.
"""

import pytest

from aether.index.analyzer import analyze_document
from aether.index.memory import MemoryIndex, build_index

DOCS = [
    {"title": "red running shoes", "brand": "nike"},
    {"title": "blue running shorts", "brand": "nike"},
    {"title": "red shoes sale", "brand": "puma"},
    {"title": "green cotton socks", "brand": "puma"},
]


@pytest.fixture
def small() -> MemoryIndex:
    return build_index(DOCS)


def test_doc_ids_are_assigned_sequentially_from_zero():
    index = MemoryIndex()
    assert [index.add(doc) for doc in DOCS] == [0, 1, 2, 3]


def test_doc_id_is_the_position_of_the_document(small):
    """Id and array position are the same number, so retrieving a hit costs an
    array index rather than a lookup table."""
    for doc_id, doc in enumerate(DOCS):
        assert small.document(doc_id) is doc


def test_posting_lists_come_out_sorted_for_free(small):
    """Nothing sorts these. Ids are handed out in increasing order as
    documents arrive, so appends are ascending by construction."""
    for term in ("red", "shoes", "nike", "puma"):
        doc_ids = small.postings(term).doc_ids
        assert doc_ids == sorted(doc_ids)


def test_builds_the_expected_inverted_index(small):
    assert small.postings("red").doc_ids == [0, 2]
    assert small.postings("running").doc_ids == [0, 1]
    assert small.postings("shoes").doc_ids == [0, 2]
    assert small.postings("nike").doc_ids == [0, 1]


def test_term_frequency_is_recorded_during_indexing():
    """Captured now although nothing reads it until BM25, because
    recomputing it later would mean a second pass over every document."""
    index = build_index([{"title": "red red red shoes"}])
    assert index.postings("red").freqs == [3]
    assert index.postings("shoes").freqs == [1]


def test_document_lengths_are_recorded(small):
    for doc_id, doc in enumerate(DOCS):
        assert small.doc_length(doc_id) == len(analyze_document(doc))


def test_counts(small):
    assert small.num_docs == 4
    assert small.num_terms == len({t for doc in DOCS for t in analyze_document(doc)})
    assert small.num_postings == sum(small.df(term) for term in small.terms())


def test_average_document_length(small):
    lengths = [len(analyze_document(doc)) for doc in DOCS]
    assert small.avg_doc_length == pytest.approx(sum(lengths) / len(lengths))


def test_an_empty_index_reports_zeroes():
    index = MemoryIndex()
    assert index.num_docs == 0
    assert index.num_terms == 0
    assert index.avg_doc_length == 0.0
    assert index.search_and("anything") == []


def test_add_all_returns_the_count():
    index = MemoryIndex()
    assert index.add_all(DOCS) == 4
