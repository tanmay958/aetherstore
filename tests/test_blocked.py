"""Tests for blocked posting lists and skipping.

Two encodings live behind one interface: short lists as plain varints, long
ones as bit-packed blocks with a skip index. Most of these run against both,
because the boundary between them is exactly where a bug would hide.
"""

import random

import pytest

from aether.index.bitpack import BLOCK_SIZE
from aether.index.blocked import (
    SMALL_LIST,
    BlockedPostings,
    encode_blocked,
    intersect_skipping,
)
from aether.index.postings import intersect


def make(count: int, spread: int = 40, seed: int = 1):
    rng = random.Random(seed)
    ids = sorted(rng.sample(range(count * spread), count))
    return ids, [rng.randint(1, 9) for _ in ids]


@pytest.mark.parametrize(
    "count",
    [0, 1, 2, SMALL_LIST - 1, SMALL_LIST, SMALL_LIST + 1, BLOCK_SIZE,
     BLOCK_SIZE + 1, 1000, 5000],
)
def test_round_trips_at_every_size_boundary(count):
    """Both encodings and the seam between them, plus block boundaries."""
    ids, freqs = make(count) if count else ([], [])
    postings = BlockedPostings(encode_blocked(ids, freqs))
    assert postings.doc_ids == ids
    assert postings.freqs == freqs
    assert postings.df == count


def test_short_lists_take_the_varint_path():
    """The median term appears in two documents, and a skip index plus two
    numpy calls for that is all overhead. Measured on identical lists, varints
    are also genuinely smaller below the threshold: 17 bytes against 19 for
    eight documents."""
    small = BlockedPostings(encode_blocked(*make(SMALL_LIST)))
    assert small._small is not None
    assert small.block_count == 1  # one notional block, so callers need no special case

    large = BlockedPostings(encode_blocked(*make(SMALL_LIST + 500)))
    assert large._small is None
    assert large.block_count > 1


def test_the_varint_path_does_not_change_any_answer(): 
    """Two encodings behind one interface, so the seam must be invisible."""
    for count in (SMALL_LIST - 1, SMALL_LIST, SMALL_LIST + 1):
        ids, freqs = make(count)
        postings = BlockedPostings(encode_blocked(ids, freqs))
        assert postings.doc_ids == ids
        assert postings.freqs == freqs
        assert postings.freq_in(ids[0]) == freqs[0]
        assert intersect_skipping(ids, postings) == ids


def test_long_lists_are_split_into_blocks():
    ids, freqs = make(1000)
    postings = BlockedPostings(encode_blocked(ids, freqs))
    assert postings.block_count == -(-1000 // BLOCK_SIZE)


def test_dense_lists_compress_hard():
    """Consecutive documents give gaps of one, which is one bit each against
    the eight a varint would spend."""
    ids = list(range(10_000, 20_000))
    encoded = encode_blocked(ids, [1] * len(ids))
    # 0.31 bytes a posting measured, against 8 at fixed width and about 2 for
    # varints.
    assert len(encoded) < len(ids) * 0.4


# --------------------------------------------------------------------------
# the skip index
# --------------------------------------------------------------------------


def test_skip_index_finds_the_right_block():
    ids, freqs = make(2000)
    postings = BlockedPostings(encode_blocked(ids, freqs))
    for position in (0, 1, 500, 1999):
        block = postings.block_containing(ids[position])
        assert block == position // BLOCK_SIZE


def test_a_target_past_the_end_reports_no_block():
    ids, freqs = make(500)
    postings = BlockedPostings(encode_blocked(ids, freqs))
    assert postings.block_containing(ids[-1] + 1) >= postings.block_count


def test_last_doc_id_is_exposed_without_decoding():
    ids, freqs = make(1000)
    assert BlockedPostings(encode_blocked(ids, freqs)).last_doc_id == ids[-1]


def test_frequency_lookup_decodes_one_block():
    ids, freqs = make(2000)
    postings = BlockedPostings(encode_blocked(ids, freqs))
    for position in (0, 700, 1999):
        assert postings.freq_in(ids[position]) == freqs[position]
    assert postings.freq_in(-1) == 0


# --------------------------------------------------------------------------
# skipping intersection
# --------------------------------------------------------------------------


def test_skipping_agrees_with_a_full_walk():
    """Skipping is an optimization, so it has to produce exactly what the
    straight walk produces."""
    rng = random.Random(20260905)
    for _ in range(200):
        long_ids, long_freqs = make(rng.randint(200, 3000), seed=rng.randint(0, 999))
        postings = BlockedPostings(encode_blocked(long_ids, long_freqs))
        short = sorted(rng.sample(range(max(long_ids) + 1), rng.randint(1, 40)))
        assert intersect_skipping(short, postings) == intersect(short, long_ids)


def test_skipping_finds_matches_scattered_across_blocks():
    ids, freqs = make(5000)
    postings = BlockedPostings(encode_blocked(ids, freqs))
    scattered = ids[::700]
    assert intersect_skipping(scattered, postings) == scattered


def test_skipping_decodes_only_the_blocks_it_needs():
    """The point. A term in a handful of documents paired with one in
    thousands should not decode the thousands."""
    ids, freqs = make(6400)  # 50 blocks
    postings = BlockedPostings(encode_blocked(ids, freqs))
    intersect_skipping(ids[:3], postings)
    assert len(postings._cache) <= 2


def test_skipping_stops_past_the_end(): 
    ids, freqs = make(500)
    postings = BlockedPostings(encode_blocked(ids, freqs))
    assert intersect_skipping([ids[-1] + 5, ids[-1] + 9], postings) == []


@pytest.mark.parametrize("candidates", [[], [1, 2, 3]])
def test_intersecting_with_an_empty_list(candidates):
    empty = BlockedPostings(encode_blocked([], []))
    assert intersect_skipping(candidates, empty) == []
