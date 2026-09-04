# AetherStore

A distributed search and streaming prediction engine, built from scratch.

An inverted index with its own on-disk segment format and compression, queried over object storage using HTTP range requests, plus a horizontally partitioned streaming inference pipeline over the same event stream.

Not a wrapper around Elasticsearch. The segment format, the postings codec, the term dictionary, the query planner, and the cost accounting are the project.

## Status

**Phase 0 complete.** The storage engine works end to end, locally and on object storage.

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

Later phases add the bloom filter and skip lists, Kafka and the indexer service, the multi-segment query coordinator, compaction, the ML pipeline, and a dashboard.
Infrastructure comes last on purpose: the segment format needs none of it, and everything downstream is a caller of it.

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
    ├── scorer.py      BM25 relevance scoring
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
