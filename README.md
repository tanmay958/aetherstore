# AetherStore

A distributed search and streaming prediction engine, built from scratch.

An inverted index with its own on-disk segment format and compression, queried over object storage using HTTP range requests, plus a horizontally partitioned streaming inference pipeline over the same event stream.

Not a wrapper around Elasticsearch. The segment format, the postings codec, the term dictionary, the query planner, and the cost accounting are the project.

## Status

**Phase 0 complete**, plus the multi-segment coordinator. Searches a real index
on Cloudflare R2.

| Step | | |
|---|---|---|
| 0.0 | Data ingestion: REES46 loader, slicer, schema | done |
| 0.1 | In-memory inverted index (the brute-force oracle) | done |
| 0.2 | Serialize and round-trip a segment to a file | done |
| 0.3 | Real 5-section layout, read only via byte ranges | done |
| 0.4 | Read counter: requests and bytes per query | done |
| 0.5 | Delta encoding and varint compression | done |
| 0.6 | BM25 scoring | done |
| 0.7 | `S3Store` against MinIO, R2 and S3 | done |
| 1.0 | Manifest, multi-segment fan-out, time pruning | done |
| 1.1 | Kafka indexer: idempotent flush, crash recovery | done |
| 1.2 | Benchmarked on 1M real REES46 events | done |
| 1.3 | Bloom filter per segment | done |
| 1.4 | Front-coded term dictionary | done |
| 1.5 | Bit-packed postings and skip lists | done |
| 4.0 | Compaction and orphan collection | done |

Later phases add the bloom filter and skip lists, Kafka and the indexer service, the multi-segment query coordinator, compaction, the ML pipeline, and a dashboard.
Infrastructure comes last on purpose: the segment format needs none of it, and everything downstream is a caller of it.

## Measured

One million real REES46 events, on an M-series laptop. Regenerate with
`uv run python -m aether.bench data/raw/2019-Oct.csv --events 1000000`.

```
INDEXING
  documents        1,000,000
  segments         100 (10,000 docs each)
  throughput       57,481 docs/sec   (single threaded, pure Python)

SIZE
  source           127.4 MB
  index             83.4 MB   0.66x of source
    postings           8.1 MB    9.6%
    docstore          67.5 MB   80.9%
    termdict           3.1 MB    3.7%
    hotcache           4.8 MB    5.8%
  bytes/posting    1.35   (8.00 at fixed width)

QUERIES (warm)               hits  requests       read      ms
  common, two terms        89,707       200   356.5 KB     230
  common, one term        386,044       100   254.4 KB     252
  rare brand                   43        28    154.0 B       7
  absent term                   0         0      0.0 B       2

TIME PRUNING "samsung smartphone"
  last hour              10/100 searched     20 requests    22 ms
  everything            100/100 searched    200 requests   225 ms

COMPACTION (merge factor 10)
                   segments   cold requests   over R2
  before                100             600      7.5 s
  after                  10              61      0.8 s
```

Cold cost is what a serverless process pays on every invocation, and it is
driven by how *many* segments there are rather than how big: opening one costs
two requests before any searching happens. Merging 100 into 10 cuts a cold
query by **10x**, and answers are identical -- same hits, same document ids,
same BM25 scores to nine decimal places, verified at a million documents.

Phase 1 took postings from 2.29 to 1.35 bytes each and the whole index from
0.75x of source to 0.66x. The docstore is now 81% of what remains, and is the
obvious next target for size.

Compaction never deletes what it retires. A coordinator that read the manifest
a moment ago is still reading those segments, so removing them is a separate
job with a grace period -- which is also why the orphan collector had to ship
in the same phase.

Two of these are worth dwelling on.

**1.35 bytes per posting**, against 8 at fixed width. Delta encoding, then bit
packing each block of 128 at the width its largest gap needs, then a varint
fallback for the 98.7% of terms that appear in fewer documents than a block
holds.

**4.3 MB of hotcache avoids reading 95 MB.** That is the whole design in one
line: a small immutable summary, fetched once and cached forever because a
segment can never change, standing in for the rest of the index.

A cold process is the state that matters for a serverless deployment, and it
is where the bloom filter earns its 0.5 MB:

```
COLD QUERIES                 hits  requests  no bloom   saved
  common, two terms        89,707       399       399      0%
  rare brand                   43        56       128     56%
  absent term                   0         1       100     99%
```

Zero for common terms, because every segment genuinely holds them. The filter
can only help when the answer is no, and the single remaining request on an
absent term is the predicted 1% false positive landing on exactly one of a
hundred segments.

## Design

**Immutable segments on object storage.**
Object stores offer no append, no lock, and no atomic rename, so nothing is ever modified. The indexer writes self-contained segment files and flips them live by rewriting one small manifest.

**Requests are the currency, not bytes.**
An S3 range GET costs 20-50 ms whether it returns 64 bytes or 8 MB, roughly 300x a local disk seek. So the segment format is built to minimize round trips rather than bytes read: a fixed-size footer at a known offset, a cacheable hotcache holding the sparse term index and a bloom filter, coalesced ranges, and parallel fan-out. Three round trips for a cold segment, one or two warm.

**Single-writer by construction.**
Segment filenames are derived from the Kafka offset range they contain, so a crash and replay overwrites a byte-identical object instead of duplicating documents. One manifest per partition means one writer per file, inherited from consumer-group assignment rather than built with consensus.

**State partitioned by the same key as the stream.**
Events are keyed by `session_id`, so every event for a session reaches exactly one predictor replica, which keeps session state in a local dict with no shared store and no locks.

## Quick start

```
uv sync
uv run pytest
```

The whole suite runs offline against a committed fixture. No dataset download, no Kaggle account, no network.

```
uv run python -m aether.data.rees46 --input tests/fixtures/rees46_sample.csv
```

```
scanned tests/fixtures/rees46_sample.csv
  rows read      30
  events         27
  rows skipped   3
    unknown event_type 'browse'                 1
    unparseable event_time                      1
    malformed row                               1
  missing brand  5 (18.5%)
  missing categ. 6 (22.2%)
  event types
    view                         16   59.3%
    add_to_cart                   6   22.2%
    purchase                      4   14.8%
    remove_from_cart              1    3.7%
```

Search it:

```
uv run python -m aether.index.search tests/fixtures/rees46_sample.csv "samsung smartphone"
```

```
index: 27 docs, 40 terms, 185 postings   built in 0.5 ms
       8.4 terms per document on average

query "samsung smartphone"  ->  ['samsung', 'smartphone']
    samsung              df      4
    smartphone           df      9
    AND                       4 docs in 0.008 ms

  doc      1  Samsung White Lite Smartphone L486    view          electronics.smartphone   $130.76
  ...
```

Build a segment and search that instead:

```
uv run python -m aether.index.build tests/fixtures/rees46_sample.csv out.seg
uv run python -m aether.index.search out.seg "samsung smartphone"
```

```
  segment size   2,638 B
    postings            410 B   15.5%
    docstore            966 B   36.6%
    termdict            972 B   36.8%
    hotcache            174 B    6.6%
    footer              116 B    4.4%

  bytes/posting  2.22  (8.00 at fixed width)
  source size    4,148 B  (0.64x)
```

The segment is smaller than the CSV it was built from, while also being a
searchable index over it and carrying derived titles the CSV does not have.

Note the hotcache: 174 bytes, read once and cached forever, which is the entire
price of never fetching the rest. And note the term dictionary, now the largest
section at 37% -- front coding it is the next obvious win.

Searching reports what it cost in requests, because on object storage the round
trip is the price, not the bytes:

```
    score     doc  title
    3.927       1  Samsung White Lite Smartphone L486   view
    3.927       4  Samsung White Lite Smartphone L486   view
    3.803       7  Samsung White Lite Smartphone L486   add_to_cart
    3.803      13  Samsung White Lite Smartphone L486   remove_from_cart

  storage cost
    open (once)      2 requests, 290 B
    term lookup      1 request, 194 B
    posting lists    2 requests, 28 B
    fetch  4 docs     1 request, 966 B
```

Ranking adds no requests: BM25 needs document frequency, which arrived with the
dictionary block, term frequency, which arrived with the postings, and document
length, which arrived with the hotcache. All three were recorded in earlier
steps for this moment.

### Many segments

An index is a pile of segments, and a manifest says which ones count. Writing a
segment does not make it searchable; naming it in the manifest does, which is
how a storage system with no transactions still gets atomic publication.

```
uv run python -m aether.index.ingest events.csv r2://aether/idx --docs-per-segment 5000
uv run python -m aether.index.search r2://aether/idx "samsung smartphone"
```

Fan-out is on a thread pool, because the engine is not CPU bound and never was.
Measured against real R2 from a laptop, four segments:

```
--workers 1     3,886 ms     20 requests
--workers 16    1,254 ms     20 requests
```

Same cost in requests and money, 3.1x the speed. Time pruning is cheaper still:
the manifest carries each segment's time span, so a query outside it discards
whole segments for **zero** requests.

### Streaming

```
make up                                                    # Redpanda + MinIO
uv run python -m aether.stream.producer events.csv         # CSV -> Kafka
uv run python -m aether.stream.indexer r2://aether         # Kafka -> segments
```

The indexer's whole correctness argument is the order of five steps:

```
1. consume offsets 100..199
2. build the index in memory
3. PUT segments/p0/...100-199.seg
4. PUT manifests/p0/current.json     <- searchable here
5. commit offset 200                 <- acknowledged only now
```

Work first, bookmark last. Committing first would be at-most-once: a crash
between the two loses the batch with nothing to detect it. This order is
at-least-once, so a crash duplicates instead, and duplication is survivable
because the segment key is derived from the offset range. A replay rewrites a
byte-identical object at the same key, so the duplicate cannot exist.

There is a test for a crash at every step, and one asserting that a replay of a
*partially* flushed batch supersedes rather than duplicates it. None of them
need a broker: offsets are just monotonic integers, which is why the indexing
core knows nothing about Kafka.

Single-writer safety is inherited rather than built. Kafka gives each partition
to exactly one consumer in a group, so one manifest per partition means one
writer per file, with no lock and no consensus algorithm.

### Scoring across segments

BM25 asks how many documents are in the collection and how many contain the
term. A segment can only answer for itself, so identical documents score
differently depending on which segment they landed in:

```
              local stats      global stats
  doc 5          3.967             3.803
  doc 1          1.772             3.927
```

`--global-stats` collects document frequencies from every segment first, so all
of them score against the same denominator. The cost is chiefly a second wave
of requests in sequence rather than more of them: document frequency lives in
the term dictionary, which scoring reads anyway. It does add a few dictionary
reads, in segments where one query term appears and another does not -- a
conjunction skips such a segment, but a corpus-wide frequency still has to
count the documents in it. Elasticsearch makes the same trade under the name
`dfs_query_then_fetch`, and defaults to local for the same reason.

The score gap above is length normalization at work -- `add_to_cart` tokenizes
into more terms than `view`, making those documents longer and therefore
slightly less relevant per match.

The same commands work against object storage, which is what `ObjectStore` was
defined for in step 3. Nothing above that layer changed to make this work:

```
uv run python -m aether.storage.check     r2://aether
uv run python -m aether.index.build       events.csv r2://aether/segments/0.seg
uv run python -m aether.index.search      r2://aether/segments/0.seg "samsung smartphone"
```

To work with real data, see [docs/DATA.md](docs/DATA.md).
For MinIO, R2, and S3 setup, see [docs/STORAGE.md](docs/STORAGE.md).

## Layout

```
src/aether/
├── events.py          canonical event schema, the boundary everything targets
├── data/
│   ├── rees46.py      streaming CSV -> canonical events (.csv and .csv.gz)
│   ├── titles.py      deterministic product titles derived from product_id
│   └── slice.py       carve a whole-session slice out of the full file
├── storage/
│   ├── base.py        ObjectStore: get_range, get_suffix, put
│   ├── local.py       filesystem-backed, stands in for S3/R2
│   ├── s3.py          AWS S3, Cloudflare R2, MinIO: one implementation
│   ├── counting.py    request and byte accounting
│   ├── factory.py     open_store / open_object from a URI
│   └── check.py       verify a backend before trusting an index to it
└── index/
    ├── analyzer.py    text -> index terms
    ├── postings.py    posting lists, intersect and union merge walks
    ├── base.py        the read interface every backend satisfies
    ├── memory.py      in-memory inverted index, the correctness oracle
    ├── codec.py       delta encoding and varints for posting lists
    ├── bloom.py       per-segment term filter, rides in the hotcache
    ├── bitpack.py     numpy bit packing at the width a block needs
    ├── blocked.py     skippable posting blocks
    ├── scorer.py      BM25 relevance scoring
    ├── manifest.py    which segments are live: the commit point
    ├── coordinator.py fan out across segments, prune, merge
    ├── ingest.py      CSV -> many segments + manifest
    ├── compactor.py   merge small segments into larger ones
    ├── gc.py          delete objects no manifest references
    ├── compact.py     CLI for both
└── stream/
    ├── config.py      Kafka connection settings
    ├── producer.py    CSV -> Kafka, keyed by session
    ├── partition.py   the indexing core: offsets in, segments out
    └── indexer.py     the consumer service
bench.py              regenerates every number below
    ├── segment.py     the 5-section binary format and its byte-range reader
    ├── build.py       CSV -> segment file
    └── search.py      query a CSV or a segment
tests/
├── fixtures/          small REES46-schema CSV, committed, all edge cases
└── test_*.py
docs/DATA.md           dataset, attribution, and what is real vs derived
```

## Data

REES46 public clickstream, ~285M events. Attribution, download steps, schema, and the precise boundary between real and derived fields are in [docs/DATA.md](docs/DATA.md).

No raw rows from the dataset are committed here; it is not under an open redistribution licence. The test fixture is hand-authored in the REES46 schema.
