"""Tests for the segment file format.

The property that matters is the round trip: an index written out and read
back must answer every query identically. Query behaviour itself is covered by
test_index_contract.py, which runs against segments as one of its backends.

The encoding tested here is deliberately temporary. Step 3 replaces the JSON
body with the five-section binary layout read through byte ranges, and these
tests should survive that change unaltered, because they assert on behaviour
rather than on bytes.
"""

import json

import pytest

from aether.data.rees46 import iter_events
from aether.index.memory import build_index
from aether.index.segment import (
    HEADER_SIZE,
    SEGMENT_MAGIC,
    SEGMENT_VERSION,
    SegmentReader,
    write_segment,
    write_segment_file,
)


@pytest.fixture
def index(sample_csv):
    return build_index(iter_events(sample_csv))


@pytest.fixture
def segment(index):
    return SegmentReader.from_bytes(write_segment(index))


# --------------------------------------------------------------------------
# the round trip
# --------------------------------------------------------------------------


def test_queries_agree_with_the_index_it_came_from(index, segment):
    """The whole point of step 2."""
    for query in (
        "samsung",
        "samsung smartphone",
        "electronics",
        "purchase",
        "bosch refrigerators",
        "nothing here",
    ):
        assert segment.search_and(query) == index.search_and(query)
        assert segment.search_or(query) == index.search_or(query)


def test_posting_lists_survive_intact(index, segment):
    assert sorted(segment.terms()) == sorted(index.terms())
    for term in index.terms():
        original, restored = index.postings(term), segment.postings(term)
        assert restored.doc_ids == original.doc_ids
        assert restored.freqs == original.freqs


def test_documents_survive_intact(index, segment):
    for doc_id in range(index.num_docs):
        assert segment.document(doc_id) == index.document(doc_id)


def test_document_lengths_survive_intact(index, segment):
    for doc_id in range(index.num_docs):
        assert segment.doc_length(doc_id) == index.doc_length(doc_id)


def test_counts_survive_intact(index, segment):
    assert segment.num_docs == index.num_docs
    assert segment.num_terms == index.num_terms
    assert segment.num_postings == index.num_postings
    assert segment.avg_doc_length == pytest.approx(index.avg_doc_length)


def test_writing_is_deterministic(index):
    """Byte-identical output for identical input. Later this is what makes a
    crashed indexer safe to replay: rewriting the same offsets produces the
    same object rather than a duplicate."""
    assert write_segment(index) == write_segment(index)


def test_a_segment_is_self_contained(index, tmp_path):
    """A segment can be searched with nothing else present. That is what lets
    any machine search any segment independently, with no coordination."""
    path = tmp_path / "standalone.seg"
    write_segment_file(index, path)
    del index

    reopened = SegmentReader.open(path)
    assert reopened.search_and("samsung smartphone") == [1, 4, 7, 13]
    assert reopened.document(1)["brand"] == "samsung"


def test_round_trips_through_a_file(index, tmp_path):
    path = tmp_path / "nested" / "out.seg"
    written = write_segment_file(index, path)
    assert written == path.stat().st_size
    assert SegmentReader.open(path).search_and("bosch") == index.search_and("bosch")


def test_round_trips_an_empty_index():
    from aether.index.memory import MemoryIndex

    segment = SegmentReader.from_bytes(write_segment(MemoryIndex()))
    assert segment.num_docs == 0
    assert segment.num_terms == 0
    assert segment.avg_doc_length == 0.0
    assert segment.search_and("anything") == []


# --------------------------------------------------------------------------
# the header
# --------------------------------------------------------------------------


def test_starts_with_magic_and_version(index):
    data = write_segment(index)
    assert data[:4] == SEGMENT_MAGIC
    assert int.from_bytes(data[4:8], "little") == SEGMENT_VERSION


def test_rejects_a_file_that_is_not_a_segment():
    with pytest.raises(ValueError, match="magic was"):
        SegmentReader.from_bytes(b"PK\x03\x04" + b"\x00" * 64)


def test_rejects_a_truncated_header():
    with pytest.raises(ValueError, match="shorter than"):
        SegmentReader.from_bytes(b"ATH")


def test_rejects_an_unsupported_version(index):
    data = bytearray(write_segment(index))
    data[4:8] = (SEGMENT_VERSION + 1).to_bytes(4, "little")
    with pytest.raises(ValueError, match="version 2 is not supported"):
        SegmentReader.from_bytes(bytes(data))


def test_rejects_a_corrupt_body(index):
    data = write_segment(index)
    with pytest.raises(ValueError, match="body is corrupt"):
        SegmentReader.from_bytes(data[: HEADER_SIZE + 20])


def test_body_is_json_for_now(index):
    """Asserting the temporary encoding on purpose: when step 3 replaces it
    with the binary layout, this test should fail and be deleted, while every
    behavioural test above keeps passing."""
    body = write_segment(index)[HEADER_SIZE:]
    assert json.loads(body)["num_docs"] == index.num_docs
