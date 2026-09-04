"""Tests for BM25.

Each test pins one of the three behaviours BM25 exists to provide, because
each corrects a specific way naive ranking goes wrong: rare terms should count
more, repetition should stop counting eventually, and long documents should
not win by being long.
"""

import pytest

from aether.index.scorer import DEFAULT_B, DEFAULT_K1, BM25

SCORER = BM25()


# --------------------------------------------------------------------------
# idf: rare terms are informative
# --------------------------------------------------------------------------


def test_rarer_terms_score_higher():
    """Matching "refrigerators" says far more than matching "view"."""
    assert SCORER.idf(1000, 1) > SCORER.idf(1000, 100) > SCORER.idf(1000, 900)


def test_idf_is_never_negative():
    """The textbook form goes negative once a term is in more than about half
    the corpus, which lets a common term subtract from a score and rank a
    document below one that matched fewer terms. Lucene's +1 inside the log
    prevents that while preserving the ordering."""
    for df in range(1, 1001):
        assert SCORER.idf(1000, df) >= 0.0


def test_a_term_in_every_document_contributes_almost_nothing():
    """Not exactly zero: the +0.5 smoothing leaves a residue. It approaches
    zero rather than reaching it, which is enough, and is why nothing in the
    scoring loop tries to short-circuit on idf == 0."""
    ubiquitous = SCORER.idf(1000, 1000)
    assert 0 < ubiquitous < 0.001
    assert ubiquitous < SCORER.idf(1000, 1) / 1000


def test_an_absent_term_contributes_nothing():
    assert SCORER.idf(1000, 0) == 0.0


# --------------------------------------------------------------------------
# tf: repetition saturates
# --------------------------------------------------------------------------


def test_more_occurrences_score_higher():
    scores = [SCORER.term_score(tf, 10, 10.0, 1.0) for tf in (1, 2, 3, 10)]
    assert scores == sorted(scores)


def test_term_frequency_saturates():
    """Fifty mentions are not fifty times more relevant than one. Without this
    the ranking rewards keyword stuffing."""
    one = SCORER.term_score(1, 10, 10.0, 1.0)
    fifty = SCORER.term_score(50, 10, 10.0, 1.0)
    assert fifty < one * 3


def test_saturation_ceiling_is_set_by_k1():
    """As tf grows the score approaches idf * (k1 + 1)."""
    huge = SCORER.term_score(10_000, 10, 10.0, 1.0)
    assert huge == pytest.approx(SCORER.k1 + 1.0, abs=0.01)


def test_k1_zero_makes_repetition_worthless():
    binary = BM25(k1=0.0)
    assert binary.term_score(1, 10, 10.0, 1.0) == binary.term_score(99, 10, 10.0, 1.0)


def test_zero_occurrences_score_zero():
    assert SCORER.term_score(0, 10, 10.0, 1.0) == 0.0


# --------------------------------------------------------------------------
# length normalization
# --------------------------------------------------------------------------


def test_shorter_documents_score_higher_for_the_same_match():
    """A long document matches more queries by accident, so relative length is
    divided out."""
    short = SCORER.term_score(2, 5, 10.0, 1.0)
    average = SCORER.term_score(2, 10, 10.0, 1.0)
    long = SCORER.term_score(2, 40, 10.0, 1.0)
    assert short > average > long


def test_b_zero_ignores_length_entirely():
    no_norm = BM25(b=0.0)
    assert no_norm.term_score(2, 5, 10.0, 1.0) == no_norm.term_score(2, 500, 10.0, 1.0)


def test_an_average_length_document_is_unaffected_by_b():
    for b in (0.0, 0.5, 1.0):
        assert BM25(b=b).term_score(3, 10, 10.0, 1.0) == pytest.approx(
            BM25(b=0.0).term_score(3, 10, 10.0, 1.0)
        )


def test_an_empty_index_does_not_divide_by_zero():
    assert SCORER.term_score(1, 0, 0.0, 1.0) > 0.0


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_defaults_match_the_literature():
    assert (SCORER.k1, SCORER.b) == (DEFAULT_K1, DEFAULT_B)
    assert (DEFAULT_K1, DEFAULT_B) == (1.2, 0.75)


@pytest.mark.parametrize("k1, b", [(-1.0, 0.75), (1.2, -0.1), (1.2, 1.5)])
def test_rejects_parameters_outside_their_range(k1, b):
    with pytest.raises(ValueError):
        BM25(k1=k1, b=b)


def test_score_scales_linearly_with_idf():
    """idf is a plain multiplier, which is what makes a term's contribution
    independent of the other terms in the query."""
    assert SCORER.term_score(3, 10, 10.0, 2.0) == pytest.approx(
        2 * SCORER.term_score(3, 10, 10.0, 1.0)
    )
