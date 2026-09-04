"""Tests for batch ingestion.

The batch ancestor of the streaming indexer: accumulate, flush a segment,
discard the memory, repeat, then publish one manifest.
"""

import pytest

from aether.data.rees46 import iter_events
from aether.index.ingest import ingest, segment_key
from aether.index.manifest import read_manifest
from aether.index.segment import SegmentReader
from aether.storage import LocalStore


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


def events(sample_csv):
    return iter_events(sample_csv)


def test_splits_into_segments_of_the_requested_size(store, sample_csv):
    manifest = ingest(events(sample_csv), store, docs_per_segment=8)
    assert [s.docs for s in manifest.segments] == [8, 8, 8, 3]
    assert manifest.docs == 27


def test_a_single_segment_when_the_batch_is_large_enough(store, sample_csv):
    manifest = ingest(events(sample_csv), store, docs_per_segment=1000)
    assert len(manifest.segments) == 1
    assert manifest.segments[0].docs == 27


def test_publishes_a_manifest_naming_every_segment(store, sample_csv):
    """Until this lands nothing is searchable; after it, everything is."""
    ingest(events(sample_csv), store, docs_per_segment=8)
    live = read_manifest(store)
    assert len(live.segments) == 4
    assert all(store.exists(s.key) for s in live.segments)


def test_segment_keys_come_from_the_row_range(store, sample_csv):
    """Deterministic naming is what will make a crashed indexer safe to
    replay: the same input rewrites the same objects rather than duplicating
    documents, and duplicates corrupt BM25 silently."""
    manifest = ingest(events(sample_csv), store, docs_per_segment=8)
    assert manifest.segments[0].key == segment_key(0, 7)
    assert manifest.segments[1].key == segment_key(8, 15)


def test_re_ingesting_the_same_input_rewrites_the_same_objects(store, sample_csv):
    first = ingest(events(sample_csv), store, docs_per_segment=8)
    before = {s.key: store.get_range(s.key, 0, s.bytes) for s in first.segments}

    second = ingest(events(sample_csv), store, docs_per_segment=8)
    after = {s.key: store.get_range(s.key, 0, s.bytes) for s in second.segments}

    assert before == after
    assert [s.key for s in first.segments] == [s.key for s in second.segments]


def test_metadata_matches_each_segment_footer(store, sample_csv):
    """The manifest duplicates footer data so a coordinator can plan and prune
    having read only the manifest."""
    manifest = ingest(events(sample_csv), store, docs_per_segment=8)
    for meta in manifest.segments:
        footer = SegmentReader(store, meta.key).footer
        assert meta.docs == footer.num_docs
        assert meta.min_ts == footer.min_ts
        assert meta.max_ts == footer.max_ts
        assert meta.bytes == store.size(meta.key)


def test_every_document_survives_exactly_once(store, sample_csv):
    manifest = ingest(events(sample_csv), store, docs_per_segment=8)
    seen = [
        SegmentReader(store, meta.key).document(i)["event_id"]
        for meta in manifest.segments
        for i in range(meta.docs)
    ]
    expected = [e["event_id"] for e in iter_events(sample_csv)]
    assert seen == expected


def test_document_ids_restart_in_every_segment(store, sample_csv):
    """Segment-local ids are what make a segment searchable with no outside
    knowledge, and why the coordinator carries (segment, doc_id) pairs."""
    manifest = ingest(events(sample_csv), store, docs_per_segment=8)
    for meta in manifest.segments:
        reader = SegmentReader(store, meta.key)
        assert reader.document(0)["event_id"]  # every segment has a doc 0


def test_an_empty_stream_publishes_an_empty_manifest(store):
    manifest = ingest(iter([]), store)
    assert manifest.segments == ()
    assert read_manifest(store).segments == ()


def test_rejects_a_nonsense_batch_size(store, sample_csv):
    with pytest.raises(ValueError, match="at least 1"):
        ingest(events(sample_csv), store, docs_per_segment=0)
