"""Tests for posting lists and the set operations over them."""

import random

import pytest

from aether.index.postings import PostingList, intersect, union


def test_records_ids_and_frequencies_in_parallel():
    posting_list = PostingList()
    posting_list.append(3, 2)
    posting_list.append(7, 1)
    assert posting_list.doc_ids == [3, 7]
    assert posting_list.freqs == [2, 1]
    assert posting_list.df == 2
    assert posting_list.freq_in(3) == 2
    assert posting_list.freq_in(99) == 0


@pytest.mark.parametrize("second", [3, 2])
def test_refuses_non_ascending_ids(second):
    """Ascending order is what delta encoding and skip pointers both rest on,
    and both fail silently if it breaks, so it is enforced here."""
    posting_list = PostingList()
    posting_list.append(3, 1)
    with pytest.raises(ValueError, match="must ascend"):
        posting_list.append(second, 1)


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
