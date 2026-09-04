"""Tests for the session-aware slicer.

The output has to be an ordinary REES46 CSV, so the loader reads a slice
through exactly the same code path as the full file. That property is what
keeps "test mode" from existing.
"""

import csv

import pytest

from aether.data.rees46 import REES46_COLUMNS, iter_events
from aether.data.slice import slice_sessions


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.reader(handle))


def test_output_is_a_valid_rees46_csv(sample_csv, tmp_path):
    dst = tmp_path / "slice.csv"
    slice_sessions(sample_csv, dst, sessions=2)
    assert read_rows(dst)[0] == list(REES46_COLUMNS)


def test_the_loader_reads_a_slice_unchanged(sample_csv, tmp_path):
    """No special case: the slice goes through the normal loader."""
    dst = tmp_path / "slice.csv"
    slice_sessions(sample_csv, dst, sessions=2)
    events = list(iter_events(dst))
    assert events
    assert all(e["session_id"] for e in events)


def test_keeps_only_the_selected_sessions(sample_csv, tmp_path):
    dst = tmp_path / "slice.csv"
    stats = slice_sessions(sample_csv, dst, sessions=2)
    assert stats.sessions_selected == 2
    assert len({e["session_id"] for e in iter_events(dst)}) == 2


def test_keeps_every_row_of_a_selected_session(sample_csv, tmp_path):
    """This is the point of slicing by session. A session cut in half can show
    an add_to_cart whose purchase was truncated away, which would turn a
    converted session into a fake abandonment."""
    full = list(iter_events(sample_csv))
    dst = tmp_path / "slice.csv"
    slice_sessions(sample_csv, dst, sessions=2)
    sliced = list(iter_events(dst))

    kept = {e["session_id"] for e in sliced}
    expected = [e for e in full if e["session_id"] in kept]
    assert [e["ts"] for e in sliced] == [e["ts"] for e in expected]


def test_asking_for_more_sessions_than_exist_keeps_everything(sample_csv, tmp_path):
    dst = tmp_path / "slice.csv"
    stats = slice_sessions(sample_csv, dst, sessions=1000)
    full = list(iter_events(sample_csv))
    assert stats.sessions_selected == len({e["session_id"] for e in full})
    assert len(list(iter_events(dst))) == len(full)


def test_slice_is_smaller_than_the_source(sample_csv, tmp_path):
    dst = tmp_path / "slice.csv"
    stats = slice_sessions(sample_csv, dst, sessions=1)
    assert 0 < stats.bytes_written < sample_csv.stat().st_size


def test_rejects_a_nonsense_session_count(sample_csv, tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        slice_sessions(sample_csv, tmp_path / "slice.csv", sessions=0)


def test_flags_when_the_scan_window_was_exhausted(sample_csv, tmp_path):
    """Hitting --scan-limit means a session may have been cut short, and the
    caller needs to know rather than silently getting broken funnels."""
    dst = tmp_path / "slice.csv"
    stats = slice_sessions(sample_csv, dst, sessions=2, scan_limit=5)
    assert stats.truncated_warning is True
