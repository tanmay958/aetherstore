"""Tests for the segment file format.

Two things are being checked. That an index survives a round trip unchanged,
which is behaviour, and that the reader gets its answers by fetching a few
small byte ranges rather than the file, which is the entire reason the format
is shaped the way it is. The second claim is worth nothing unasserted, so a
counting store wraps the real one and the read pattern is pinned exactly.

Query behaviour is covered in test_index_contract.py, which runs against
segments as one of its backends.
"""

import pytest
from aether.data.rees46 import iter_events
from aether.index.memory import MemoryIndex, build_index
from aether.index.segment import (
    DOCS_PER_BLOCK,
    FOOTER_SIZE,
    SEGMENT_MAGIC,
    SEGMENT_VERSION,
    TERMS_PER_BLOCK,
    Footer,
    SegmentReader,
    write_segment,
    write_segment_file,
)
from aether.storage import CountingStore, LocalStore


@pytest.fixture
def index(sample_csv):
    return build_index(iter_events(sample_csv))


@pytest.fixture
def store(tmp_path, index):
    inner = LocalStore(tmp_path)
    inner.put("s.seg", write_segment(index))
    return CountingStore(inner)


@pytest.fixture
def segment(store):
    return SegmentReader(store, "s.seg")


@pytest.fixture
def big_index():
    """Large enough to span several dictionary and docstore blocks, which the
    27-document fixture does not."""
    docs = [
        {
            "title": f"widget{i} shared item{i % 7}",
            "brand": f"brand{i % 13}",
            "ts": 1_700_000_000 + i,
        }
        for i in range(200)
    ]
    return build_index(docs)


# --------------------------------------------------------------------------
# the round trip
# --------------------------------------------------------------------------


def test_queries_agree_with_the_index_it_came_from(index, segment):
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


def test_survives_spanning_many_blocks(big_index, tmp_path):
    """The 27-document fixture fits in one docstore block and two dictionary
    blocks, so block boundaries go untested without this."""
    store = LocalStore(tmp_path)
    store.put("big.seg", write_segment(big_index))
    segment = SegmentReader(store, "big.seg")

    assert len(segment._term_blocks) > 1
    assert len(segment._doc_blocks) == -(-200 // DOCS_PER_BLOCK)
    assert all(len(b) <= TERMS_PER_BLOCK for b in segment._dict_cache.values())

    for doc_id in (0, DOCS_PER_BLOCK - 1, DOCS_PER_BLOCK, 199):
        assert segment.document(doc_id) == big_index.document(doc_id)
    for term in ("widget0", "widget199", "shared", "brand7"):
        assert segment.postings(term).doc_ids == big_index.postings(term).doc_ids


def test_writing_is_deterministic(index):
    """Byte-identical output for identical input. This is what makes a crashed
    indexer safe to replay later: rewriting the same Kafka offsets produces
    the same object rather than a duplicate."""
    assert write_segment(index) == write_segment(index)


def test_a_segment_is_self_contained(index, tmp_path):
    """A segment can be searched with nothing else present, which is what lets
    any machine search any segment independently, with no coordination."""
    path = tmp_path / "standalone.seg"
    write_segment_file(index, path)
    del index

    reopened = SegmentReader.open(path)
    assert reopened.search_and("samsung smartphone") == [1, 4, 7, 13]
    assert reopened.document(1)["brand"] == "samsung"


def test_round_trips_an_empty_index(tmp_path):
    store = LocalStore(tmp_path)
    store.put("empty.seg", write_segment(MemoryIndex()))
    segment = SegmentReader(store, "empty.seg")

    assert segment.num_docs == 0
    assert segment.num_terms == 0
    assert segment.avg_doc_length == 0.0
    assert segment.search_and("anything") == []
    assert segment.postings("anything") is None


# --------------------------------------------------------------------------
# the footer
# --------------------------------------------------------------------------


def test_footer_is_magic_at_both_ends(index):
    data = write_segment(index)
    assert data[-FOOTER_SIZE : -FOOTER_SIZE + 4] == SEGMENT_MAGIC
    assert data[-4:] == SEGMENT_MAGIC


def test_footer_is_found_from_the_tail_without_knowing_the_size(index):
    """The whole trick: `Range: bytes=-116` needs no prior knowledge of the
    file, not even its length, and locates every other section."""
    data = write_segment(index)
    footer = Footer.unpack(data[-FOOTER_SIZE:])
    assert footer.num_docs == index.num_docs
    assert footer.version == SEGMENT_VERSION


def test_footer_sections_tile_the_file_without_gaps(index):
    data = write_segment(index)
    f = Footer.unpack(data[-FOOTER_SIZE:])
    assert f.postings_offset == 0
    assert f.docstore_offset == f.postings_offset + f.postings_length
    assert f.termdict_offset == f.docstore_offset + f.docstore_length
    assert f.hotcache_offset == f.termdict_offset + f.termdict_length
    assert f.hotcache_offset + f.hotcache_length + FOOTER_SIZE == len(data)


def test_footer_records_the_time_span(index, segment):
    """Cheapest optimization available: a query filtered to the last hour can
    discard a segment whose newest event is days old, for zero requests."""
    timestamps = [index.document(i)["ts"] for i in range(index.num_docs)]
    assert segment.footer.min_ts == min(timestamps)
    assert segment.footer.max_ts == max(timestamps)


def test_rejects_a_file_that_is_not_a_segment():
    with pytest.raises(ValueError, match="footer magic"):
        Footer.unpack(b"\x00" * FOOTER_SIZE)


def test_rejects_something_shorter_than_a_footer():
    with pytest.raises(ValueError, match="shorter than"):
        Footer.unpack(b"ATHR")


def test_rejects_an_unsupported_version(index):
    data = bytearray(write_segment(index))
    data[-FOOTER_SIZE + 4 : -FOOTER_SIZE + 8] = (SEGMENT_VERSION + 1).to_bytes(4, "little")
    with pytest.raises(ValueError, match=f"version {SEGMENT_VERSION + 1} is not supported"):
        Footer.unpack(bytes(data))


# --------------------------------------------------------------------------
# the read pattern, which is the point of the format
# --------------------------------------------------------------------------


def test_opening_costs_two_requests(store):
    """Footer, then the hotcache it points at. Nothing else."""
    SegmentReader(store, "s.seg")
    assert store.stats.requests == 2


def test_opening_reads_a_tiny_fraction_of_the_file(store):
    SegmentReader(store, "s.seg")
    assert store.stats.bytes_read < store.size("s.seg") * 0.25


def test_a_term_costs_one_dictionary_read_and_one_postings_read(store, segment):
    store.reset()
    segment.postings("samsung")
    assert store.stats.requests == 2


def test_the_dictionary_block_is_cached_after_first_use(store, segment):
    """Segments are immutable, so a fetched block can be kept forever with no
    invalidation logic at all."""
    segment.postings("samsung")
    store.reset()
    segment.postings("samsung")
    assert store.stats.requests == 1  # postings only; the dictionary block was cached


def test_document_frequency_costs_no_postings_read(store, segment):
    """df lives in the dictionary precisely so BM25 never pays a request for
    it, and never fetches a whole posting list just to count entries."""
    segment.df("samsung")  # warms the dictionary block
    store.reset()
    assert segment.df("samsung") == 4
    assert store.stats.requests == 0


def test_an_absent_term_costs_at_most_one_read(store, segment):
    store.reset()
    assert segment.postings("helicopter") is None
    assert store.stats.requests <= 1


def test_documents_in_one_block_cost_one_read(store, segment):
    store.reset()
    segment.document(0)
    segment.document(1)
    segment.document(2)
    assert store.stats.requests == 1


def test_a_whole_query_costs_a_handful_of_requests(store, segment):
    """Two terms and four displayed documents: one dictionary block, two
    posting lists, one docstore block."""
    store.reset()
    hits = segment.search_and("samsung smartphone")
    for doc_id in hits:
        segment.document(doc_id)

    assert hits == [1, 4, 7, 13]
    assert store.stats.requests <= 6


def test_bytes_read_is_bounded_by_block_size_not_file_size(big_index, tmp_path):
    """On the 27-document fixture a query still moves most of the file, and
    that is honest rather than a failure: every document lives in a single
    docstore block, so fetching one fetches all of them.

    Read amplification is bounded by block size, not by file size, which only
    becomes visible once a segment holds more than one block. With 200
    documents across four blocks a query touches a fraction, and a real
    10,000 document segment spans 157 blocks.
    """
    inner = LocalStore(tmp_path)
    inner.put("big.seg", write_segment(big_index))
    store = CountingStore(inner)

    segment = SegmentReader(store, "big.seg")
    store.reset()
    hits = segment.search_and("widget7")
    for doc_id in hits:
        segment.document(doc_id)

    assert hits
    assert store.stats.bytes_read < store.size("big.seg") * 0.35


def test_the_reader_never_fetches_the_whole_file(store):
    """The claim the format exists to support."""
    segment = SegmentReader(store, "s.seg")
    segment.search_and("samsung smartphone")
    assert store.stats.bytes_read < store.size("s.seg")
