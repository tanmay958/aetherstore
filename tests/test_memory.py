"""Tests for the in-memory inverted index.

This index is the oracle every later stage is checked against, so it gets
checked against something simpler still: a brute-force linear scan. If the
dict and the scan agree on randomized queries, the dict is trustworthy enough
to judge the segment reader by.
"""

import random

import pytest

from aether.data.rees46 import iter_events
from aether.index.analyzer import analyze_document, tokenize
from aether.index.memory import MemoryIndex, PostingList, build_index, intersect, union

DOCS = [
    {"title": "red running shoes", "brand": "nike"},
    {"title": "blue running shorts", "brand": "nike"},
    {"title": "red shoes sale", "brand": "puma"},
    {"title": "green cotton socks", "brand": "puma"},
]


@pytest.fixture
def small() -> MemoryIndex:
    return build_index(DOCS)


@pytest.fixture
def fixture_index(sample_csv) -> MemoryIndex:
    return build_index(iter_events(sample_csv))


def brute_force_and(index: MemoryIndex, query: str) -> list[int]:
    """The dumbest correct implementation: scan every document."""
    wanted = set(tokenize(query))
    if not wanted:
        return []
    return [
        doc_id
        for doc_id in range(index.num_docs)
        if wanted <= set(analyze_document(index.document(doc_id)))
    ]


def brute_force_or(index: MemoryIndex, query: str) -> list[int]:
    wanted = set(tokenize(query))
    if not wanted:
        return []
    return [
        doc_id
        for doc_id in range(index.num_docs)
        if wanted & set(analyze_document(index.document(doc_id)))
    ]


# --------------------------------------------------------------------------
# PostingList
# --------------------------------------------------------------------------


def test_posting_list_records_ids_and_frequencies_in_parallel():
    posting_list = PostingList()
    posting_list.append(3, 2)
    posting_list.append(7, 1)
    assert posting_list.doc_ids == [3, 7]
    assert posting_list.freqs == [2, 1]
    assert posting_list.df == 2
    assert posting_list.freq_in(3) == 2
    assert posting_list.freq_in(99) == 0


@pytest.mark.parametrize("second", [3, 2])
def test_posting_list_refuses_non_ascending_ids(second):
    """Ascending order is what delta encoding and skip pointers both rest on,
    so it is enforced rather than assumed."""
    posting_list = PostingList()
    posting_list.append(3, 1)
    with pytest.raises(ValueError, match="must ascend"):
        posting_list.append(second, 1)


# --------------------------------------------------------------------------
# intersect / union
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([0, 2], [0, 2], [0, 2]),
        ([0, 1, 2], [1], [1]),
        ([0, 2, 4], [1, 3, 5], []),
        ([], [1, 2], []),
        ([1, 2], [], []),
        ([1, 5, 9], [5, 9, 13], [5, 9]),
    ],
)
def test_intersect(a, b, expected):
    assert intersect(a, b) == expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([0, 2], [0, 2], [0, 2]),
        ([0, 2, 4], [1, 3], [0, 1, 2, 3, 4]),
        ([], [1, 2], [1, 2]),
        ([1, 2], [], [1, 2]),
    ],
)
def test_union(a, b, expected):
    assert union(a, b) == expected


def test_merge_walks_agree_with_sets_on_random_input():
    """The merge walk exists because the segment reader cannot use sets. It
    still has to produce exactly what sets would."""
    rng = random.Random(20260904)
    for _ in range(400):
        a = sorted(rng.sample(range(120), rng.randint(0, 40)))
        b = sorted(rng.sample(range(120), rng.randint(0, 40)))
        assert intersect(a, b) == sorted(set(a) & set(b))
        assert union(a, b) == sorted(set(a) | set(b))


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------


def test_doc_ids_are_assigned_sequentially_from_zero():
    index = MemoryIndex()
    assert [index.add(doc) for doc in DOCS] == [0, 1, 2, 3]


def test_doc_id_is_the_position_of_the_document(small):
    """Id and array position are the same number, so retrieving a hit costs an
    index rather than a lookup table."""
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


def test_document_frequency(small):
    assert small.df("red") == 2
    assert small.df("socks") == 1
    assert small.df("absent") == 0


def test_term_frequency_is_recorded(small):
    index = build_index([{"title": "red red red shoes"}])
    assert index.postings("red").freqs == [3]
    assert index.postings("shoes").freqs == [1]


def test_unknown_term_has_no_posting_list(small):
    assert small.postings("helicopter") is None


# --------------------------------------------------------------------------
# searching
# --------------------------------------------------------------------------


def test_and_returns_only_documents_containing_every_term(small):
    assert small.search_and("red shoes") == [0, 2]
    assert small.search_and("red running") == [0]


def test_and_is_empty_when_any_term_is_absent(small):
    """One missing term collapses the whole conjunction, with no walking."""
    assert small.search_and("red helicopter") == []


def test_or_returns_documents_containing_any_term(small):
    assert small.search_or("socks shorts") == [1, 3]


@pytest.mark.parametrize("query", ["", "   ", "the of and", "a"])
def test_queries_with_no_indexable_terms_return_nothing(small, query):
    assert small.search_and(query) == []
    assert small.search_or(query) == []


def test_results_are_returned_in_ascending_document_order(small):
    """Ranking arrives with BM25 in step 6. Until then, document order."""
    hits = small.search_or("red running shoes")
    assert hits == sorted(hits)


def test_repeating_a_term_does_not_change_the_result(small):
    assert small.search_and("red red shoes") == small.search_and("red shoes")


def test_term_order_does_not_change_the_result(small):
    """Posting lists are intersected shortest-first, so the walk order differs
    from the query order. The answer must not."""
    assert small.search_and("red shoes") == small.search_and("shoes red")


# --------------------------------------------------------------------------
# the oracle check
# --------------------------------------------------------------------------


def test_and_matches_a_brute_force_scan_on_real_data(fixture_index):
    rng = random.Random(20260904)
    vocabulary = [term for term, _ in fixture_index.most_common_terms(40)]
    for _ in range(300):
        query = " ".join(rng.sample(vocabulary, rng.randint(1, 3)))
        assert fixture_index.search_and(query) == brute_force_and(fixture_index, query)


def test_or_matches_a_brute_force_scan_on_real_data(fixture_index):
    rng = random.Random(4711)
    vocabulary = [term for term, _ in fixture_index.most_common_terms(40)]
    for _ in range(300):
        query = " ".join(rng.sample(vocabulary, rng.randint(1, 3)))
        assert fixture_index.search_or(query) == brute_force_or(fixture_index, query)


def test_queries_mixing_known_and_unknown_terms(fixture_index):
    rng = random.Random(99)
    vocabulary = [term for term, _ in fixture_index.most_common_terms(20)]
    for _ in range(100):
        query = f"{rng.choice(vocabulary)} zzz{rng.randint(0, 999)}"
        assert fixture_index.search_and(query) == brute_force_and(fixture_index, query)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def test_counts(small):
    assert small.num_docs == 4
    assert small.num_terms == len({t for doc in DOCS for t in analyze_document(doc)})
    assert small.num_postings == sum(
        small.postings(term).df for term in small._postings
    )


def test_average_document_length(small):
    lengths = [len(analyze_document(doc)) for doc in DOCS]
    assert small.avg_doc_length == pytest.approx(sum(lengths) / len(lengths))


def test_an_empty_index_reports_zeroes():
    index = MemoryIndex()
    assert index.num_docs == 0
    assert index.num_terms == 0
    assert index.avg_doc_length == 0.0
    assert index.search_and("anything") == []


def test_most_common_terms_are_ordered_by_document_frequency(fixture_index):
    counts = [df for _, df in fixture_index.most_common_terms(10)]
    assert counts == sorted(counts, reverse=True)


def test_indexes_the_whole_fixture(fixture_index):
    assert fixture_index.num_docs == 27
    assert fixture_index.search_and("samsung smartphone") == [1, 4, 7, 13]
