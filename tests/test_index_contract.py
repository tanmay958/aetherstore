"""Behaviour every index implementation must share.

These run against each backend in turn via the `backend` fixture: the
in-memory dict today, a segment file today, a byte-range reader over object
storage later. Adding a backend to that fixture subjects it to this entire
suite at once.

The oracle tests at the bottom are the reason the in-memory index exists.
Randomized queries are answered by both the index under test and a
brute-force linear scan, and the two must agree exactly. When a compressed,
lazily-decoded reader disagrees with a loop over every document, the loop is
right.
"""

import random

import pytest

from aether.index.analyzer import analyze_document, tokenize


def brute_force_and(index, query: str) -> list[int]:
    """The dumbest correct implementation: look at every document."""
    wanted = set(tokenize(query))
    if not wanted:
        return []
    return [
        doc_id
        for doc_id in range(index.num_docs)
        if wanted <= set(analyze_document(index.document(doc_id)))
    ]


def brute_force_or(index, query: str) -> list[int]:
    wanted = set(tokenize(query))
    if not wanted:
        return []
    return [
        doc_id
        for doc_id in range(index.num_docs)
        if wanted & set(analyze_document(index.document(doc_id)))
    ]


# --------------------------------------------------------------------------
# lookups
# --------------------------------------------------------------------------


def test_reports_the_shape_of_the_index(backend):
    assert backend.num_docs == 27
    assert backend.num_terms == 40
    assert backend.num_postings == 185
    assert backend.avg_doc_length == pytest.approx(8.44, abs=0.01)


def test_postings_count_documents_while_lengths_count_occurrences(backend):
    """These two numbers differ, and the gap is not a bug.

    A term occurring twice in one document is a single posting but two terms
    of document length. Here the brand appears in both the `brand` field and
    the derived title, so 228 term occurrences collapse into 185 postings.
    The posting count is what drives segment size; the length is what BM25
    normalizes by.
    """
    occurrences = sum(backend.doc_length(i) for i in range(backend.num_docs))
    assert occurrences > backend.num_postings


def test_document_frequency_matches_the_posting_list(backend):
    for term in backend.terms():
        assert backend.df(term) == backend.postings(term).df


def test_unknown_term_has_no_posting_list(backend):
    assert backend.postings("helicopter") is None
    assert backend.df("helicopter") == 0


def test_posting_lists_are_ascending(backend):
    for term in backend.terms():
        doc_ids = backend.postings(term).doc_ids
        assert doc_ids == sorted(doc_ids)
        assert len(set(doc_ids)) == len(doc_ids)


def test_every_posting_points_at_a_real_document(backend):
    for term in backend.terms():
        for doc_id in backend.postings(term).doc_ids:
            assert 0 <= doc_id < backend.num_docs


def test_documents_are_retrievable_by_id(backend):
    for doc_id in range(backend.num_docs):
        assert backend.document(doc_id)["event_id"] == f"r_{doc_id + 1:09d}"


def test_document_lengths_are_available(backend):
    for doc_id in range(backend.num_docs):
        assert backend.doc_length(doc_id) == len(
            analyze_document(backend.document(doc_id))
        )


# --------------------------------------------------------------------------
# querying
# --------------------------------------------------------------------------


def test_and_requires_every_term(backend):
    assert backend.search_and("samsung smartphone") == [1, 4, 7, 13]


def test_and_is_empty_when_any_term_is_absent(backend):
    """One missing term collapses the conjunction with no walking at all."""
    assert backend.search_and("samsung helicopter") == []


def test_or_requires_only_one_term(backend):
    hits = backend.search_or("samsung bosch")
    assert set(backend.search_and("samsung")) <= set(hits)
    assert set(backend.search_and("bosch")) <= set(hits)


@pytest.mark.parametrize("query", ["", "   ", "the of and", "a"])
def test_queries_with_no_indexable_terms_return_nothing(backend, query):
    assert backend.search_and(query) == []
    assert backend.search_or(query) == []


def test_results_are_in_ascending_document_order(backend):
    """Ranking arrives with BM25 in step 6. Until then, document order."""
    for query in ("samsung smartphone", "electronics", "view purchase"):
        hits = backend.search_or(query)
        assert hits == sorted(hits)


def test_repeating_a_term_changes_nothing(backend):
    assert backend.search_and("samsung samsung") == backend.search_and("samsung")


def test_term_order_changes_nothing(backend):
    """Posting lists are intersected shortest-first, so the walk order differs
    from the query order. The answer must not."""
    assert backend.search_and("samsung smartphone") == backend.search_and(
        "smartphone samsung"
    )


def test_case_is_ignored(backend):
    assert backend.search_and("SAMSUNG SmartPhone") == backend.search_and(
        "samsung smartphone"
    )


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------


def test_ranked_search_returns_the_same_documents_as_matching(backend):
    """Scoring changes the order, never the membership."""
    result = backend.search("samsung smartphone", top_k=100)
    assert sorted(hit.doc_id for hit in result.hits) == backend.search_and(
        "samsung smartphone"
    )


def test_results_come_back_best_first(backend):
    scores = [hit.score for hit in backend.search("electronics", top_k=20, mode="or")]
    assert scores == sorted(scores, reverse=True)


def test_total_counts_all_matches_not_just_the_page(backend):
    """Returned rather than recomputed, because matching twice would fetch
    every posting list twice."""
    result = backend.search("electronics", top_k=2, mode="or")
    assert len(result.hits) == 2
    assert result.total == len(backend.search_or("electronics"))


def test_top_k_limits_the_page(backend):
    for k in (1, 3, 100):
        assert len(backend.search("electronics", top_k=k, mode="or").hits) <= k


def test_a_rarer_term_outranks_a_common_one(backend):
    """The whole point of idf. "bosch" appears in a handful of documents,
    "view" in most of them, so a bosch match should win."""
    result = backend.search("bosch view", top_k=20, mode="or")
    ranked = [hit.doc_id for hit in result.hits]
    bosch_docs = set(backend.search_and("bosch"))
    assert ranked[0] in bosch_docs


def test_ties_break_by_document_id_so_results_are_stable(backend):
    result = backend.search("samsung smartphone", top_k=10)
    pairs = [(-hit.score, hit.doc_id) for hit in result.hits]
    assert pairs == sorted(pairs)


def test_scores_are_identical_across_backends(sample_csv, tmp_path, backend):
    """A byte-range reader over compressed postings must produce exactly the
    same floats as a plain dict, or ranking silently depends on storage."""
    from aether.index.memory import build_index
    from aether.data.rees46 import iter_events

    reference = build_index(iter_events(sample_csv))
    for query in ("samsung smartphone", "electronics", "bosch", "view purchase"):
        expected = reference.search(query, top_k=20, mode="or")
        actual = backend.search(query, top_k=20, mode="or")
        assert [h.doc_id for h in actual.hits] == [h.doc_id for h in expected.hits]
        assert [h.score for h in actual.hits] == [h.score for h in expected.hits]


@pytest.mark.parametrize("query", ["", "the of and", "helicopter"])
def test_unmatchable_queries_rank_nothing(backend, query):
    result = backend.search(query)
    assert result.hits == []
    assert result.total == 0


def test_rejects_an_unknown_mode(backend):
    with pytest.raises(ValueError, match="mode must be"):
        backend.search("samsung", mode="maybe")


# --------------------------------------------------------------------------
# the oracle
# --------------------------------------------------------------------------


def test_and_matches_a_brute_force_scan(backend):
    rng = random.Random(20260904)
    vocabulary = [term for term, _ in backend.most_common_terms(40)]
    for _ in range(300):
        query = " ".join(rng.sample(vocabulary, rng.randint(1, 3)))
        assert backend.search_and(query) == brute_force_and(backend, query)


def test_or_matches_a_brute_force_scan(backend):
    rng = random.Random(4711)
    vocabulary = [term for term, _ in backend.most_common_terms(40)]
    for _ in range(300):
        query = " ".join(rng.sample(vocabulary, rng.randint(1, 3)))
        assert backend.search_or(query) == brute_force_or(backend, query)


def test_queries_mixing_known_and_unknown_terms(backend):
    rng = random.Random(99)
    vocabulary = [term for term, _ in backend.most_common_terms(20)]
    for _ in range(100):
        query = f"{rng.choice(vocabulary)} zzz{rng.randint(0, 999)}"
        assert backend.search_and(query) == brute_force_and(backend, query)
