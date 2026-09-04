"""Tests for the canonical event schema."""

import pytest

from aether.events import EVENT_FIELDS, make_event, validate_event


def _event(**overrides) -> dict:
    base = dict(
        event_id="r_000000001",
        ts=1569888001,
        session_id="s1",
        user_id="u1",
        event_type="view",
    )
    return make_event(**{**base, **overrides})


def test_key_order_matches_the_schema():
    """The segment writer depends on a stable field order, so this is not a
    cosmetic property."""
    assert tuple(_event()) == EVENT_FIELDS


def test_absent_fields_default_to_none_rather_than_being_omitted():
    event = _event()
    assert event["device"] is None
    assert event["title"] is None
    assert event["query"] is None


def test_rejects_an_unknown_event_type():
    with pytest.raises(ValueError, match="unknown event_type"):
        _event(event_type="browse")


def test_validate_accepts_a_well_formed_event():
    validate_event(_event())


def test_validate_rejects_missing_keys():
    event = _event()
    del event["price"]
    with pytest.raises(ValueError, match="do not match schema"):
        validate_event(event)


def test_validate_rejects_reordered_keys():
    """Same keys, wrong order. A dict comparison would pass; the writer would
    not."""
    event = _event()
    reordered = {key: event[key] for key in reversed(EVENT_FIELDS)}
    with pytest.raises(ValueError, match="do not match schema"):
        validate_event(reordered)


def test_validate_rejects_a_non_integer_timestamp():
    event = _event()
    event["ts"] = "1569888001"
    with pytest.raises(ValueError, match="ts must be an int"):
        validate_event(event)
