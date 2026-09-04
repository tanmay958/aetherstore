"""Tests for bit packing.

A codec that is wrong loses data silently, so these lean on randomized round
trips rather than hand-picked examples. One case is called out by name: an
early version applied numpy's `byteswap()` after a big-endian view, which
reverses the bytes while leaving the dtype claiming big-endian, so values came
back scrambled. That version was timed and never checked, which is exactly how
a fast wrong codec gets shipped.
"""

import numpy as np
import pytest

from aether.index.bitpack import BLOCK_SIZE, pack, packed_size, unpack, width_for


@pytest.mark.parametrize("bits", range(1, 33))
def test_round_trips_at_every_width(bits):
    ceiling = (1 << bits) - 1
    values = np.array([0, 1, ceiling // 2, ceiling], dtype=np.uint32)
    assert np.array_equal(unpack(pack(values, bits), len(values), bits), values)


def test_round_trips_random_values():
    rng = np.random.default_rng(20260905)
    for _ in range(300):
        count = int(rng.integers(1, 400))
        values = rng.integers(0, 1 << int(rng.integers(1, 25)), count).astype(np.uint32)
        bits = width_for(values)
        assert np.array_equal(unpack(pack(values, bits), count, bits), values)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([0], 1),
        ([1], 1),
        ([2], 2),
        ([255], 8),
        ([256], 9),
        ([2**31], 32),
        ([1, 1, 1, 1000], 10),  # the widest value sets the width
    ],
)
def test_width_is_set_by_the_largest_value(values, expected):
    assert width_for(np.array(values, dtype=np.uint32)) == expected


def test_width_of_nothing_is_one():
    """Never zero: a width of zero would pack no bits and decode nothing."""
    assert width_for(np.zeros(0, dtype=np.uint32)) == 1


def test_empty_input_packs_to_nothing():
    assert pack(np.zeros(0, dtype=np.uint32), 4) == b""
    assert unpack(b"", 0, 4).size == 0


def test_predicted_size_matches_reality():
    rng = np.random.default_rng(7)
    for _ in range(100):
        count = int(rng.integers(1, 300))
        bits = int(rng.integers(1, 33))
        values = rng.integers(0, 1 << bits, count).astype(np.uint32) >> 0
        assert len(pack(values, bits)) == packed_size(count, bits)


def test_dense_gaps_cost_one_bit_each():
    """The case delta encoding produces constantly: a term in consecutive
    documents gives a run of gaps all equal to one."""
    gaps = np.ones(BLOCK_SIZE, dtype=np.uint32)
    assert width_for(gaps) == 1
    assert len(pack(gaps, 1)) == BLOCK_SIZE // 8  # 16 bytes for 128 values


def test_beats_a_byte_per_value_on_small_numbers():
    """Against the varints this replaces, which spend a whole byte on a one."""
    values = np.random.default_rng(3).integers(0, 8, 1024).astype(np.uint32)
    assert len(pack(values, width_for(values))) < len(values) * 0.5
