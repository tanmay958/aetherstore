"""Tests for the posting list codec.

A codec that is wrong loses data silently, so these lean on round trips over
randomized input rather than on hand-picked examples.
"""

import random

import pytest

from aether.index.codec import (
    decode_varint,
    decode_varints,
    delta_decode,
    delta_encode,
    encode_varint,
    encode_varints,
)


@pytest.mark.parametrize(
    "value, size",
    [
        (0, 1), (1, 1), (127, 1),          # one byte up to 7 bits
        (128, 2), (16_383, 2),             # two bytes up to 14
        (16_384, 3), (2_097_151, 3),
        (2_097_152, 4),
        (2**32 - 1, 5),
    ],
)
def test_varint_width_follows_magnitude(value, size):
    """The whole point: small numbers cost less. After delta encoding most
    gaps land in the one-byte range."""
    assert len(encode_varint(value)) == size


def test_varint_round_trip_over_random_values():
    rng = random.Random(20260904)
    for _ in range(2000):
        value = rng.randint(0, 2**40)
        assert decode_varint(encode_varint(value)) == (value, len(encode_varint(value)))


def test_varints_round_trip_as_a_run():
    rng = random.Random(11)
    values = [rng.randint(0, 2**20) for _ in range(500)]
    data = encode_varints(values)
    decoded, pos = decode_varints(data, len(values))
    assert decoded == values
    assert pos == len(data)


def test_varints_decode_from_an_offset():
    """Runs sit back to back inside a posting list, so decoding has to resume
    at an arbitrary position."""
    data = encode_varints([1, 2, 3]) + encode_varints([400, 500])
    first, pos = decode_varints(data, 3)
    second, pos = decode_varints(data, 2, pos)
    assert first == [1, 2, 3]
    assert second == [400, 500]
    assert pos == len(data)


def test_empty_run():
    assert encode_varints([]) == b""
    assert decode_varints(b"", 0) == ([], 0)


def test_rejects_negative_values():
    with pytest.raises(ValueError, match="unsigned"):
        encode_varint(-1)
    with pytest.raises(ValueError, match="unsigned"):
        encode_varints([1, -2])


def test_rejects_a_truncated_varint():
    with pytest.raises(ValueError, match="truncated"):
        decode_varint(b"\x80\x80")


def test_delta_encoding_shrinks_the_numbers():
    assert delta_encode([1000, 1003, 1007, 1100]) == [1000, 3, 4, 93]


def test_delta_round_trip():
    rng = random.Random(7)
    for _ in range(300):
        values = sorted(rng.sample(range(200_000), rng.randint(1, 200)))
        assert delta_decode(delta_encode(values)) == values


@pytest.mark.parametrize("values", [[], [0], [5]])
def test_delta_edge_cases(values):
    assert delta_decode(delta_encode(values)) == values


def test_delta_rejects_non_ascending_input():
    """A repeat or a step backwards would encode a negative gap and corrupt
    the list without complaint."""
    with pytest.raises(ValueError, match="ascending"):
        delta_encode([1, 5, 5])
    with pytest.raises(ValueError, match="ascending"):
        delta_encode([9, 2])


def test_consecutive_documents_cost_one_byte_each():
    """The case that matters most. A common term appears in run after run of
    adjacent documents, and every one of those gaps is 1."""
    values = list(range(1000, 2000))
    encoded = encode_varints(delta_encode(values))
    assert len(encoded) == 2 + 999  # a 2-byte first id, then 999 single bytes


def test_beats_fixed_width_on_realistic_postings():
    rng = random.Random(3)
    values = sorted(rng.sample(range(100_000), 5_000))
    compressed = len(encode_varints(delta_encode(values)))
    assert compressed < 4 * len(values) * 0.5
