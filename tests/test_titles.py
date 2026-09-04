"""Tests for derived product titles.

The property that matters most is stability. A title must be identical whether
it is computed today over 100k rows or next month over 10M, in any process, on
any machine. Everything else is secondary to that.
"""

import os
import subprocess
import sys

from aether.data.titles import derive_title


def test_same_product_always_gets_the_same_title():
    first = derive_title("1005105", "samsung", "electronics.smartphone")
    second = derive_title("1005105", "samsung", "electronics.smartphone")
    assert first == second


def test_title_is_stable_across_processes():
    """The real guard against using the builtin hash(), which is salted per
    process by PYTHONHASHSEED and would hand out a different title on every
    run. CRC32 is not affected, so all three subprocesses must agree.
    """
    code = (
        "from aether.data.titles import derive_title; "
        "print(derive_title('1005105', 'samsung', 'electronics.smartphone'))"
    )
    outputs = set()
    for seed in ("0", "1", "987654"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1, f"title changed with hash seed: {outputs}"


def test_title_does_not_depend_on_what_was_titled_before():
    """No shared counter, no RNG stream: order of calls must not matter."""
    derive_title.cache_clear()
    direct = derive_title("1801690", "bosch", "appliances.kitchen.refrigerators")

    derive_title.cache_clear()
    for product in ("1", "2", "3", "4", "5"):
        derive_title(product, "other", "misc.thing")
    after_others = derive_title("1801690", "bosch", "appliances.kitchen.refrigerators")

    assert direct == after_others


def test_real_brand_and_category_leaf_appear_verbatim():
    """Only the adjectives are invented; the anchors come from the file."""
    title = derive_title("1005105", "samsung", "electronics.smartphone")
    assert title.startswith("Samsung ")
    assert "Smartphone" in title


def test_uses_only_the_category_leaf_not_the_whole_path():
    title = derive_title("1801690", "bosch", "appliances.kitchen.refrigerators")
    assert "Refrigerators" in title
    assert "Appliances" not in title


def test_underscores_in_a_leaf_become_spaces():
    title = derive_title("42", "lg", "appliances.kitchen.washing_machine")
    assert "Washing Machine" in title


def test_missing_brand_still_works_when_a_category_exists():
    title = derive_title("16000870", None, "furniture.living_room.sofa")
    assert title is not None
    assert "Sofa" in title


def test_missing_category_still_works_when_a_brand_exists():
    title = derive_title("17300353", "creed", None)
    assert title is not None
    assert title.startswith("Creed ")


def test_returns_none_when_there_is_no_real_anchor():
    """With neither brand nor category, a title would be pure fabrication.
    Decline instead of inventing one."""
    assert derive_title("100067795", None, None) is None
    assert derive_title("100067795", "", "") is None


def test_different_products_mostly_get_different_titles():
    titles = {derive_title(str(i), "acme", "electronics.smartphone") for i in range(500)}
    # Colour, modifier, and a model code of 26,000 combinations: collisions are
    # possible but should be rare.
    assert len(titles) > 450


def test_derived_vocabulary_is_skewed_not_uniform():
    """BM25 ranks a rare term above a common one, so the derived adjectives
    must vary in frequency the way real product text does."""
    from collections import Counter

    colors = Counter(
        derive_title(str(i), "acme", "electronics.smartphone").split()[1]
        for i in range(4000)
    )
    counts = sorted(colors.values(), reverse=True)
    assert counts[0] > 5 * counts[-1]
