"""Shared test fixtures.

The sample CSV is committed and tiny, so the whole suite runs offline with no
Kaggle account and no 1.6 GB download. It is hand-authored in the exact REES46
schema and deliberately contains the awkward cases the real file contains:
missing brands, missing category codes, a zero price, an unrecognised event
type, an unparseable timestamp, and a truncated row.
"""

import gzip
import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# Counts for tests/fixtures/rees46_sample.csv, kept here so a change to the
# fixture fails loudly in one place rather than silently across many tests.
SAMPLE_GOOD_ROWS = 27
SAMPLE_BAD_ROWS = 3
SAMPLE_SESSIONS = 7


@pytest.fixture
def sample_csv() -> Path:
    """A small REES46-format CSV, uncompressed."""
    return FIXTURES / "rees46_sample.csv"


@pytest.fixture
def sample_csv_gz(sample_csv: Path, tmp_path: Path) -> Path:
    """The same fixture, gzipped, to exercise the compressed read path."""
    target = tmp_path / "rees46_sample.csv.gz"
    with sample_csv.open("rb") as src, gzip.open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return target
