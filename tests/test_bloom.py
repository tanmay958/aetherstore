"""Tests for the bloom filter.

One property matters above all others: a "no" must never be wrong. A false
"maybe" costs a wasted dictionary read; a false "no" would silently drop
matching documents from a result, which is the kind of bug that looks like bad
relevance rather than a bug.
"""

import random
import string

import pytest

from aether.index.bloom import DEFAULT_FALSE_POSITIVE_RATE, BloomFilter, optimal_size


def words(n: int, seed: int = 1, alphabet: str = string.ascii_lowercase) -> list[str]:
    rng = random.Random(seed)
    return list({"".join(rng.choices(alphabet, k=8)) for _ in range(n)})


def test_never_denies_a_term_it_contains():
    """The only guarantee the design relies on."""
    terms = words(5000)
    filter_ = BloomFilter.for_terms(terms)
    assert all(term in filter_ for term in terms)


def test_false_positive_rate_is_near_the_target():
    terms = words(4000, seed=1)
    filter_ = BloomFilter.for_terms(terms, DEFAULT_FALSE_POSITIVE_RATE)

    absent = words(40_000, seed=2, alphabet=string.ascii_uppercase)
    positives = sum(1 for term in absent if term in filter_)
    rate = positives / len(absent)
    assert rate < DEFAULT_FALSE_POSITIVE_RATE * 2


def test_two_hash_halves_are_independent():
    """The first attempt used CRC32 under two different seeds, whose outputs
    are linearly related, and measured 1.6% against a 1.0% target. Splitting
    one BLAKE2b digest fixed it. A regression here would be silent, showing up
    only as more wasted reads."""
    terms = words(4000, seed=7)
    filter_ = BloomFilter.for_terms(terms, 0.01)
    absent = words(40_000, seed=8, alphabet=string.ascii_uppercase)
    rate = sum(1 for term in absent if term in filter_) / len(absent)
    assert rate < 0.015


def test_is_stable_across_processes():
    """Built in one run, queried in the next. The builtin hash() is salted per
    process and would make a persisted filter answer differently every time."""
    import os
    import subprocess
    import sys

    code = (
        "from aether.index.bloom import BloomFilter;"
        "f = BloomFilter.for_terms(['samsung', 'smartphone', 'bosch']);"
        "print(f.to_bytes().hex())"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("0", "1", "424242")
    }
    assert len(outputs) == 1


def test_round_trips_through_bytes():
    terms = words(1000)
    original = BloomFilter.for_terms(terms)
    restored, end = BloomFilter.from_bytes(original.to_bytes())

    assert end == len(original.to_bytes())
    assert restored.bits == original.bits
    assert restored.hashes == original.hashes
    assert all(term in restored for term in terms)


def test_decodes_from_an_offset():
    """It rides at the end of the hotcache, after the document lengths."""
    filter_ = BloomFilter.for_terms(["samsung"])
    padded = b"\x00" * 17 + filter_.to_bytes()
    restored, _ = BloomFilter.from_bytes(padded, 17)
    assert "samsung" in restored


@pytest.mark.parametrize("n", [1, 10, 1000, 100_000])
def test_sizing_scales_with_term_count(n):
    bits, hashes = optimal_size(n, 0.01)
    assert bits % 8 == 0
    assert bits / n >= 9  # roughly 9.6 bits per term at 1%
    assert 1 <= hashes <= 12


def test_a_tighter_target_costs_more_bits():
    loose, _ = optimal_size(1000, 0.1)
    tight, _ = optimal_size(1000, 0.001)
    assert tight > loose


def test_an_empty_filter_denies_everything():
    filter_ = BloomFilter.for_terms([])
    assert "anything" not in filter_


def test_load_factor_reports_saturation():
    """Far above one half means the filter is undersized for its contents and
    false positives will be common."""
    terms = words(4000)
    assert 0.3 < BloomFilter.for_terms(terms).load < 0.7

    crammed = BloomFilter(64, 7)
    for term in terms[:200]:
        crammed.add(term)
    assert crammed.load > 0.9


def test_rejects_a_nonsense_rate():
    with pytest.raises(ValueError, match="false positive rate"):
        optimal_size(100, 1.5)


def test_rejects_a_partial_byte():
    with pytest.raises(ValueError, match="whole number of bytes"):
        BloomFilter(bits=13, hashes=3)
