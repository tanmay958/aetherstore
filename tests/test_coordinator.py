"""Tests for the multi-segment coordinator.

The claim being tested is that splitting an index across segments changes how
it is stored and not what it answers. The strongest form of that is the
sharding invariance test below: the same documents in one segment and in four
must produce identical scores, provided the scoring uses corpus-wide
statistics.
"""

import pytest

from aether.data.rees46 import iter_events
from aether.index.coordinator import Coordinator
from aether.index.ingest import ingest
from aether.index.manifest import Manifest, SegmentMeta, write_manifest
from aether.index.memory import build_index
from aether.storage import CountingStore, LocalStore


@pytest.fixture
def sharded(tmp_path, sample_csv):
    """The fixture split across four segments."""
    store = CountingStore(LocalStore(tmp_path))
    ingest(iter_events(sample_csv), store, docs_per_segment=8)
    store.reset()
    return Coordinator(store)


@pytest.fixture
def whole(tmp_path, sample_csv):
    """The same documents in a single segment."""
    store = LocalStore(tmp_path / "whole")
    ingest(iter_events(sample_csv), store, docs_per_segment=10_000)
    return Coordinator(store)


@pytest.fixture
def reference(sample_csv):
    """The in-memory oracle, still the arbiter of correctness."""
    return build_index(iter_events(sample_csv))


def event_ids(coordinator, result):
    return [coordinator.document(hit)["event_id"] for hit in result.hits]


# --------------------------------------------------------------------------
# correctness across segments
# --------------------------------------------------------------------------


def test_finds_every_match_across_segments(sharded, reference):
    result = sharded.search("samsung smartphone", top_k=100)
    assert result.total == len(reference.search_and("samsung smartphone"))


def test_returns_the_same_documents_as_the_oracle(sharded, reference):
    result = sharded.search("samsung smartphone", top_k=100)
    expected = {
        reference.document(doc_id)["event_id"]
        for doc_id in reference.search_and("samsung smartphone")
    }
    assert set(event_ids(sharded, result)) == expected


def test_or_mode_spans_segments(sharded, reference):
    result = sharded.search("bosch samsung", top_k=100, mode="or")
    assert result.total == len(reference.search_or("bosch samsung"))


def test_sharding_does_not_change_scores_under_global_stats(sharded, whole, reference):
    """The invariant that matters most.

    Four segments and one segment must rank identically, because how the data
    was chopped up is a storage decision and relevance is not.
    """
    query = "samsung smartphone"
    from_shards = sharded.search(query, top_k=10, global_stats=True)
    from_whole = whole.search(query, top_k=10, global_stats=True)

    assert event_ids(sharded, from_shards) == event_ids(whole, from_whole)
    assert [round(h.score, 9) for h in from_shards.hits] == [
        round(h.score, 9) for h in from_whole.hits
    ]

    # And both agree with the plain in-memory index.
    expected = reference.search(query, top_k=10)
    assert [round(h.score, 9) for h in from_shards.hits] == [
        round(h.score, 9) for h in expected.hits
    ]


def test_local_stats_do_shift_scores_between_segments(sharded, whole):
    """The honest cost of the default. A term rare in a small segment and
    common in a large one earns different scores for identical documents,
    which is why global_stats exists and why Elasticsearch offers the same
    choice under the name dfs_query_then_fetch."""
    query = "samsung smartphone"
    local = [round(h.score, 6) for h in sharded.search(query, top_k=10).hits]
    unified = [round(h.score, 6) for h in whole.search(query, top_k=10).hits]
    assert local != unified


def test_global_stats_costs_a_second_wave(sharded):
    """The price is chiefly latency, not requests.

    No postings fetches are added: document frequency lives in the term
    dictionary, which scoring reads anyway, and those blocks are cached before
    the second pass. A few dictionary reads can be added, in segments where
    some query terms appear and others do not. A conjunction skips such a
    segment, but a corpus-wide document frequency still has to count the
    documents in it, so paying to look is inherent to the correct answer.
    """
    query = "samsung smartphone"
    terms = 2
    sharded.search(query, top_k=10)  # warm the dictionary blocks

    sharded.store.reset()
    sharded.search(query, top_k=10)
    one_wave = sharded.store.stats.requests

    sharded.store.reset()
    two_waves = sharded.search(query, top_k=10, global_stats=True)

    assert two_waves.stats.waves == 2
    # Bounded by one dictionary block per term per segment, and nowhere near
    # the doubling that re-fetching postings would cost.
    assert one_wave <= sharded.store.stats.requests <= one_wave + terms * len(
        sharded.manifest.segments
    )


# --------------------------------------------------------------------------
# pruning
# --------------------------------------------------------------------------


def test_time_pruning_discards_segments_for_zero_requests(sharded):
    """The cheapest optimization available: arithmetic on the manifest."""
    segments = sharded.manifest.segments
    latest = max(s.max_ts for s in segments)

    sharded.store.reset()
    result = sharded.search("smartphone", top_k=10, start=latest + 1)

    assert result.stats.segments_pruned == len(segments)
    assert result.stats.segments_searched == 0
    assert result.hits == []
    assert sharded.store.stats.requests == 0


def test_a_narrow_window_prunes_some_but_not_all(sharded):
    first = sharded.manifest.segments[0]
    result = sharded.search("smartphone", top_k=10, start=first.min_ts, end=first.max_ts)
    assert 0 < result.stats.segments_searched < result.stats.segments_total


def test_no_window_searches_everything(sharded):
    result = sharded.search("smartphone", top_k=50)
    assert result.stats.segments_pruned == 0
    assert result.stats.segments_searched == result.stats.segments_total


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


def test_hits_carry_their_segment(sharded):
    """Document ids are segment-local, so a bare id is meaningless."""
    for hit in sharded.search("smartphone", top_k=10, mode="or").hits:
        assert hit.segment in {s.key for s in sharded.manifest.segments}


def test_results_are_ordered_best_first(sharded):
    scores = [h.score for h in sharded.search("smartphone", top_k=50, mode="or").hits]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_the_page_but_not_the_count(sharded):
    result = sharded.search("smartphone", top_k=2, mode="or")
    assert len(result.hits) == 2
    assert result.total > 2


def test_ordering_is_stable_across_runs(sharded):
    """Segments are searched on a thread pool, so completion order varies.
    Ties break on segment then document id to keep output deterministic."""
    runs = [
        [(h.segment, h.doc_id) for h in sharded.search("smartphone", top_k=20, mode="or").hits]
        for _ in range(5)
    ]
    assert all(run == runs[0] for run in runs)


def test_unmatchable_query_returns_nothing(sharded):
    result = sharded.search("helicopter")
    assert result.hits == []
    assert result.total == 0


# --------------------------------------------------------------------------
# the live set
# --------------------------------------------------------------------------


def test_an_index_with_no_manifest_is_empty(tmp_path):
    coordinator = Coordinator(LocalStore(tmp_path))
    assert coordinator.manifest.segments == ()
    assert coordinator.search("anything").hits == []


def test_a_segment_not_in_the_manifest_is_invisible(sharded):
    """Uploading a segment does not publish it. The manifest does."""
    kept = sharded.manifest.segments[:2]
    write_manifest(sharded.store, Manifest().with_segments(kept))
    sharded.refresh()

    result = sharded.search("smartphone", top_k=50, mode="or")
    assert result.stats.segments_total == 2
    assert {h.segment for h in result.hits} <= {s.key for s in kept}


def test_refresh_picks_up_new_segments(sharded):
    before = sharded.search("smartphone", top_k=50, mode="or").total

    trimmed = sharded.manifest.segments[:1]
    write_manifest(sharded.store, Manifest().with_segments(trimmed))
    sharded.refresh()
    fewer = sharded.search("smartphone", top_k=50, mode="or").total

    assert fewer < before


def test_readers_are_opened_once_and_kept(sharded):
    """Opening costs two requests for a footer and a hotcache, and neither can
    ever change, so a coordinator that has served one query has already paid
    for every segment it touched."""
    sharded.search("smartphone", top_k=10, mode="or")
    sharded.store.reset()
    sharded.search("smartphone", top_k=10, mode="or")

    # No re-opening: only dictionary and postings reads remain.
    assert sharded.store.stats.requests < 2 * len(sharded.manifest.segments)


def test_stale_readers_are_dropped_on_refresh(sharded):
    write_manifest(sharded.store, Manifest().with_segments(sharded.manifest.segments[:1]))
    sharded.refresh()
    assert set(sharded._readers) <= {s.key for s in sharded.manifest.segments}


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------


def test_documents_are_fetched_for_the_displayed_hits(sharded):
    result = sharded.search("smartphone", top_k=5, mode="or")
    docs = sharded.documents(result.hits)
    assert len(docs) == len(result.hits)
    assert all(doc["event_id"] for doc in docs)


def test_fetching_nothing_reads_nothing(sharded):
    sharded.store.reset()
    assert sharded.documents([]) == []
    assert sharded.store.stats.requests == 0
