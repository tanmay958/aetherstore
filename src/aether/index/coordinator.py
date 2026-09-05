"""Querying an index made of many segments.

A single segment answers questions about the documents inside it. An index is
a pile of segments, and this is the thing that turns one into the other: read
the manifest, discard the segments that cannot match, ask the rest in
parallel, and merge what comes back.

It stores nothing. Every byte it works from lives in object storage, and the
only state it keeps is a cache of footers and hotcaches that can never go
stale because segments never change. That is what makes it possible to run
several of these behind a load balancer, or on Lambda, with no coordination
between them at all.

Three things make a query cheap, in increasing order of how much they save.

**Pruning.** The manifest carries each segment's time span, so a query limited
to the last hour discards everything older using nothing but arithmetic on
data already in memory. Zero requests for each segment eliminated.

**Parallel fan-out.** A range read against R2 measured from a laptop takes
roughly 200 milliseconds, and almost all of that is waiting rather than
working. Twenty segments in sequence is four seconds; twenty at once is about
a quarter of one. The engine is not CPU bound and never was, so threads are
the right tool and a thread pool is the whole implementation.

**Caching.** Opening a segment costs two requests, and both are cacheable
forever. A coordinator that has served one query has already paid for every
segment it touched.

## Scoring across segments

BM25 asks how many documents are in the collection and how many contain the
term. A segment can only answer for itself, and those answers are wrong for an
index made of many. A term in three documents of a small segment looks rare
there and common in a large one, so identical documents score differently
depending on where they happened to land.

Both options are here, because the trade is real and every production engine
makes the same one.

`global_stats=False`, the default, lets each segment score with its own
numbers. One wave of requests, and rankings that are slightly inconsistent
between segments. Elasticsearch defaults to exactly this.

`global_stats=True` collects document frequencies from every segment first,
then scores everything against the same denominator. Correct, and the cost is
chiefly a second round trip in sequence, so at 200 milliseconds a wave the
query takes twice as long. Elasticsearch calls this `dfs_query_then_fetch`.

It adds no *postings* reads, because document frequency comes from the term
dictionary, which scoring has to read anyway, and those blocks are cached
before the second pass touches them. It can add dictionary reads, in segments
where some query terms appear and others do not: a conjunction skips such a
segment entirely, but a corpus-wide document frequency still has to count the
documents in it. Paying to look at segments that will never contribute a
result is inherent to the correct answer, not an implementation detail.
"""

from __future__ import annotations

import heapq
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from aether.index.analyzer import tokenize
from aether.index.base import CorpusStats
from aether.index.manifest import (
    DEFAULT_MANIFEST_KEY,
    PARTITION_MANIFEST_PREFIX,
    Manifest,
    SegmentMeta,
    partition_manifest_keys,
    read_manifest,
    union_manifests,
)
from aether.index.scorer import DEFAULT_SCORER, BM25
from aether.index.segment import SegmentReader
from aether.storage.base import ObjectStore


@dataclass(frozen=True)
class GlobalHit:
    """A result, identified by segment as well as document.

    Document ids are local to a segment: doc 0 of one is unrelated to doc 0 of
    another. That is deliberate, since it is what makes a segment searchable
    with no outside knowledge, and it means a coordinator must carry the pair
    around and only resolve it to a document at the very end, for the handful
    actually being displayed.
    """

    segment: str
    doc_id: int
    score: float


@dataclass
class QueryStats:
    """What a query cost, in the terms that matter on object storage."""

    segments_total: int = 0
    segments_pruned: int = 0
    segments_searched: int = 0
    elapsed_ms: float = 0.0
    waves: int = 1

    def __str__(self) -> str:
        return (
            f"{self.segments_searched}/{self.segments_total} segments searched, "
            f"{self.segments_pruned} pruned, {self.waves} wave(s), "
            f"{self.elapsed_ms:.0f} ms"
        )


@dataclass
class QueryResult:
    hits: list[GlobalHit] = field(default_factory=list)
    total: int = 0
    stats: QueryStats = field(default_factory=QueryStats)

    def __len__(self) -> int:
        return len(self.hits)

    def __iter__(self):
        return iter(self.hits)


class Coordinator:
    """Search every live segment of an index."""

    def __init__(
        self,
        store: ObjectStore,
        *,
        manifest_key: str = DEFAULT_MANIFEST_KEY,
        manifest_prefix: str | None = None,
        max_workers: int = 16,
    ) -> None:
        self.store = store
        self.manifest_key = manifest_key
        # When set, the live set is every per-partition manifest under this
        # prefix rather than one object. That is what a streamed index looks
        # like: the indexer writes one manifest per Kafka partition, because
        # exactly one consumer owns a partition and therefore exactly one
        # process writes that manifest. Searching the whole index means
        # reading all of them.
        self.manifest_prefix = manifest_prefix
        self.max_workers = max_workers
        self._manifest: Manifest | None = None
        self._readers: dict[str, SegmentReader] = {}

    # -- the live set ------------------------------------------------------

    @property
    def manifest(self) -> Manifest:
        if self._manifest is None:
            self._manifest = self._read_live_set()
        return self._manifest

    def _read_live_set(self) -> Manifest:
        if self.manifest_prefix is None:
            return read_manifest(self.store, self.manifest_key)

        keys = partition_manifest_keys(self.store, self.manifest_prefix)
        if not keys:
            return Manifest()
        # Read together, for the same reason segments are searched together:
        # this is a dozen sequential round trips otherwise, paid on every
        # cold start.
        if len(keys) == 1:
            return read_manifest(self.store, keys[0])
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(keys))) as pool:
            return union_manifests(
                pool.map(lambda key: read_manifest(self.store, key), keys)
            )

    def refresh(self) -> Manifest:
        """Re-read the manifest, picking up segments written since.

        Readers are kept, because a segment named in both the old and new
        manifest is byte for byte the same object. Immutability means a
        refresh costs one request and invalidates nothing.
        """
        self._manifest = self._read_live_set()
        live = {segment.key for segment in self._manifest.segments}
        self._readers = {k: v for k, v in self._readers.items() if k in live}
        return self._manifest

    def reader(self, key: str) -> SegmentReader:
        """A reader for one segment, opened at most once.

        The two requests that opening costs buy a footer and a hotcache, and
        neither can ever change, so they are held for the coordinator's life.
        """
        reader = self._readers.get(key)
        if reader is None:
            reader = self._readers[key] = SegmentReader(self.store, key)
        return reader

    @property
    def num_docs(self) -> int:
        return self.manifest.docs

    # -- querying ----------------------------------------------------------

    def candidates(
        self, start: int | None = None, end: int | None = None
    ) -> list[SegmentMeta]:
        """Segments that could match a time window.

        The cheapest thing in the engine: whole segments discarded on data
        already in memory, for no requests at all.
        """
        return [s for s in self.manifest.segments if s.overlaps(start, end)]

    def _map(self, segments: list[SegmentMeta], fn) -> list:
        """Run `fn` over segments concurrently.

        Threads rather than asyncio because the work is a blocking HTTP call
        per segment and boto3 is thread safe. Almost all of the elapsed time
        is spent waiting on the network, so the pool converts a sum of
        latencies into a maximum of them.
        """
        if not segments:
            return []
        if len(segments) == 1:
            return [fn(segments[0])]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(segments))) as pool:
            return list(pool.map(fn, segments))

    def _gather_corpus_stats(
        self, segments: list[SegmentMeta], query: str
    ) -> CorpusStats:
        """Collection-wide document frequencies, in one wave of requests.

        Adds no postings fetches: document frequency lives in the term
        dictionary, and the blocks read here are cached before the scoring
        pass asks for postings from them. It can add dictionary reads in
        segments the query will not match, because a corpus-wide frequency
        must count documents that never appear in a result.
        """
        terms = sorted(set(tokenize(query)))

        def dfs_for(segment: SegmentMeta) -> dict[str, int]:
            reader = self.reader(segment.key)
            return {term: reader.df(term) for term in terms}

        totals = dict.fromkeys(terms, 0)
        for partial in self._map(segments, dfs_for):
            for term, df in partial.items():
                totals[term] += df

        docs = sum(segment.docs for segment in segments)
        lengths = sum(
            self.reader(segment.key).footer.total_length for segment in segments
        )
        return CorpusStats(docs, lengths / docs if docs else 0.0, totals)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        offset: int = 0,
        mode: str = "and",
        start: int | None = None,
        end: int | None = None,
        global_stats: bool = False,
        scorer: BM25 = DEFAULT_SCORER,
    ) -> QueryResult:
        """Search every live segment and merge the best results.

        `offset` skips that many of the best hits, which is how a second page
        is fetched. It is not free, and it is worth being precise about why.

        A distributed index cannot start at rank 50 without first knowing what
        ranks 1 to 49 are, and no segment knows that alone: any of them might
        hold the top hit. So page two costs the whole of page one as well.
        Every engine that fans out has this property, and every one of them
        caps the depth rather than pretending otherwise.

        The cost is in merging, not in reading. Each segment still returns at
        most `offset + top_k` hits, and the extra ones are integers being
        sorted, not documents being fetched. Only the page actually returned
        is turned into documents.
        """
        if offset < 0:
            raise ValueError(f"offset must not be negative, got {offset}")
        began = time.perf_counter()
        all_segments = self.manifest.segments
        surviving = self.candidates(start, end)

        stats = QueryStats(
            segments_total=len(all_segments),
            segments_pruned=len(all_segments) - len(surviving),
            segments_searched=len(surviving),
        )

        if not surviving:
            stats.elapsed_ms = (time.perf_counter() - began) * 1000
            return QueryResult(stats=stats)

        corpus = None
        if global_stats:
            corpus = self._gather_corpus_stats(surviving, query)
            stats.waves = 2

        # The best `offset + top_k` overall cannot contain more than that many
        # from any single segment, so that is what each is asked for.
        depth = offset + top_k

        def search_one(segment: SegmentMeta):
            reader = self.reader(segment.key)
            return segment.key, reader.search(
                query, top_k=depth, mode=mode, scorer=scorer, corpus=corpus
            )

        merged: list[GlobalHit] = []
        total = 0
        for key, result in self._map(surviving, search_one):
            total += result.total
            merged.extend(GlobalHit(key, hit.doc_id, hit.score) for hit in result.hits)

        # Ties fall back to segment key then document id, so a repeated query
        # returns the same order regardless of which thread finished first.
        ranked = heapq.nsmallest(
            depth, merged, key=lambda hit: (-hit.score, hit.segment, hit.doc_id)
        )
        stats.elapsed_ms = (time.perf_counter() - began) * 1000
        return QueryResult(ranked[offset:], total, stats)

    # -- fetching ----------------------------------------------------------

    def document(self, hit: GlobalHit) -> dict:
        return self.reader(hit.segment).document(hit.doc_id)

    def documents(self, hits: list[GlobalHit]) -> list[dict]:
        """Fetch the displayed documents, one docstore block read per segment.

        Done concurrently for the same reason the search is: this is the last
        wave of requests in a query, and it should cost one round trip rather
        than one per result.
        """
        if not hits:
            return []
        by_segment: dict[str, list[int]] = {}
        for hit in hits:
            by_segment.setdefault(hit.segment, []).append(hit.doc_id)

        def warm(key: str) -> None:
            reader = self.reader(key)
            for doc_id in by_segment[key]:
                reader.document(doc_id)

        keys = list(by_segment)
        if len(keys) > 1:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(keys))) as pool:
                list(pool.map(warm, keys))
        return [self.document(hit) for hit in hits]
