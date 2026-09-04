"""Tests for the REES46 streaming loader.

Expected counts for the committed fixture live in conftest. The fixture also
carries three deliberately bad rows, because tolerating them is a requirement:
a loader that raises on row 12,000,003 of a 40M row file is useless.
"""

import inspect
from pathlib import Path

import pytest
from conftest import SAMPLE_GOOD_ROWS, SAMPLE_SESSIONS

from aether.data.rees46 import LoadStats, iter_events, parse_timestamp
from aether.events import validate_event

# 2019-10-01 00:00:00 UTC
OCT_1_2019 = 1569888000


def load(path: Path, **kwargs) -> tuple[list[dict], LoadStats]:
    stats = LoadStats()
    events = list(iter_events(path, stats=stats, **kwargs))
    return events, stats


# --------------------------------------------------------------------------
# timestamps
# --------------------------------------------------------------------------


def test_parses_the_published_timestamp_format():
    assert parse_timestamp("2019-10-01 00:00:00 UTC") == OCT_1_2019
    assert parse_timestamp("2019-10-01 00:00:01 UTC") == OCT_1_2019 + 1
    assert parse_timestamp("2019-10-01 01:02:03 UTC") == OCT_1_2019 + 3723


def test_day_caching_does_not_change_results():
    """The date cache is a speed optimization; it must not alter behaviour
    when the same day is parsed repeatedly or days are interleaved."""
    a = parse_timestamp("2019-10-01 12:00:00 UTC")
    b = parse_timestamp("2019-10-02 12:00:00 UTC")
    assert parse_timestamp("2019-10-01 12:00:00 UTC") == a
    assert b - a == 86400


def test_rejects_a_malformed_timestamp():
    with pytest.raises(ValueError):
        parse_timestamp("not-a-timestamp")


# --------------------------------------------------------------------------
# streaming behaviour
# --------------------------------------------------------------------------


def test_iter_events_is_a_generator():
    """Non-negotiable: the same call has to work on a 5.5 GB file."""
    assert inspect.isgeneratorfunction(iter_events)


def test_reads_gzip_and_plain_identically(sample_csv, sample_csv_gz):
    plain, _ = load(sample_csv)
    gzipped, _ = load(sample_csv_gz)
    assert plain == gzipped


def test_rejects_a_csv_that_is_not_rees46(tmp_path):
    other = tmp_path / "other.csv"
    other.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match="does not look like a REES46 CSV"):
        list(iter_events(other))


def test_handles_an_empty_file(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert list(iter_events(empty)) == []


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------


def test_every_event_matches_the_canonical_schema(sample_csv):
    events, _ = load(sample_csv)
    for event in events:
        validate_event(event)


def test_emits_one_event_per_good_row(sample_csv):
    events, stats = load(sample_csv)
    assert len(events) == SAMPLE_GOOD_ROWS
    assert stats.events == SAMPLE_GOOD_ROWS


def test_cart_is_renamed_to_add_to_cart(sample_csv):
    _, stats = load(sample_csv)
    assert "cart" not in stats.event_types
    assert stats.event_types == {
        "view": 16,
        "add_to_cart": 6,
        "purchase": 4,
        "remove_from_cart": 1,
    }


def test_event_ids_are_contiguous(sample_csv):
    events, _ = load(sample_csv)
    assert [e["event_id"] for e in events] == [
        f"r_{i:09d}" for i in range(1, SAMPLE_GOOD_ROWS + 1)
    ]


def test_timestamps_become_int_epochs(sample_csv):
    events, _ = load(sample_csv)
    assert events[0]["ts"] == OCT_1_2019 + 1
    assert all(isinstance(e["ts"], int) for e in events)


def test_tracks_the_time_span(sample_csv):
    _, stats = load(sample_csv)
    assert stats.min_ts == OCT_1_2019 + 1
    assert stats.max_ts == OCT_1_2019 + 238  # 00:03:58


def test_fields_rees46_lacks_are_none(sample_csv):
    """REES46 has no device column and no search events. Those stay empty
    rather than being filled with a plausible-looking guess."""
    events, _ = load(sample_csv)
    assert all(e["device"] is None for e in events)
    assert all(e["query"] is None for e in events)


def test_sessions_are_preserved(sample_csv):
    events, _ = load(sample_csv)
    assert len({e["session_id"] for e in events}) == SAMPLE_SESSIONS


# --------------------------------------------------------------------------
# tolerating real-world gaps
# --------------------------------------------------------------------------


def test_bad_rows_are_skipped_and_counted_not_raised(sample_csv):
    _, stats = load(sample_csv)
    assert stats.rows_skipped == 3
    assert stats.skip_reasons["malformed row"] == 1
    assert stats.skip_reasons["unparseable event_time"] == 1
    assert stats.skip_reasons["unknown event_type 'browse'"] == 1


def test_empty_brand_becomes_none(sample_csv):
    events, stats = load(sample_csv)
    assert stats.missing_brand == 5
    assert all(e["brand"] != "" for e in events)


def test_empty_category_becomes_none(sample_csv):
    events, stats = load(sample_csv)
    assert stats.missing_category == 6
    assert all(e["category"] != "" for e in events)


def test_prices_are_floats_and_zero_is_kept(sample_csv):
    events, _ = load(sample_csv)
    assert all(isinstance(e["price"], float) for e in events)
    assert any(e["price"] == 0.0 for e in events)


# --------------------------------------------------------------------------
# derived titles
# --------------------------------------------------------------------------


def test_titles_are_derived_by_default(sample_csv):
    events, stats = load(sample_csv)
    titled = [e for e in events if e["title"]]
    assert titled
    # Only the product with neither brand nor category goes untitled.
    assert stats.missing_title == 2


def test_the_same_product_is_titled_consistently_within_a_load(sample_csv):
    events, _ = load(sample_csv)
    by_product: dict[str, set[str]] = {}
    for event in events:
        by_product.setdefault(event["product_id"], set()).add(event["title"])
    assert all(len(titles) == 1 for titles in by_product.values())


def test_titles_can_be_turned_off(sample_csv):
    events, _ = load(sample_csv, derive_titles=False)
    assert all(e["title"] is None for e in events)


# --------------------------------------------------------------------------
# limit
# --------------------------------------------------------------------------


@pytest.mark.parametrize("limit", [1, 5, 27])
def test_limit_stops_early(sample_csv, limit):
    events, stats = load(sample_csv, limit=limit)
    assert len(events) == limit
    assert stats.events == limit


def test_limit_beyond_the_file_returns_everything(sample_csv):
    events, _ = load(sample_csv, limit=10_000)
    assert len(events) == SAMPLE_GOOD_ROWS


def test_limit_is_a_prefix_of_an_unlimited_read(sample_csv):
    everything, _ = load(sample_csv)
    first_five, _ = load(sample_csv, limit=5)
    assert first_five == everything[:5]
