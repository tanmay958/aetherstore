# AetherStore

A distributed search and streaming prediction engine, built from scratch.

An inverted index with its own on-disk segment format and compression, queried over object storage using HTTP range requests, plus a horizontally partitioned streaming inference pipeline over the same event stream.

Not a wrapper around Elasticsearch. The segment format, the postings codec, the term dictionary, the query planner, and the cost accounting are the project.

## Status

Phase 0, step 3 of 8: **byte-range reads**.

| Step | | |
|---|---|---|
| 0.0 | Data ingestion: REES46 loader, slicer, schema | done |
| 0.1 | In-memory inverted index (the brute-force oracle) | done |
| 0.2 | Serialize and round-trip a segment to a file | done |
| 0.3 | Real 5-section layout, read only via byte ranges | done |
| 0.4 | Read counter: requests and bytes per query | next |
| 0.5 | Delta encoding and varint compression | |
| 0.6 | BM25 scoring | |
| 0.7 | `S3Store` against MinIO, one config line | |

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
  segment size   10,866 B
    postings          1,640 B   15.1%
    docstore          7,964 B   73.3%
    termdict            972 B    8.9%
    hotcache            174 B    1.6%
    footer              116 B    1.1%
```

Note the hotcache: 174 bytes, read once and cached forever, which is the entire
price of never fetching the other 98%. And note the docstore at 73% -- still raw
JSON, which is what step 5 is for.

To work with real data, see [docs/DATA.md](docs/DATA.md).

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
│   └── local.py       filesystem-backed, stands in for S3/R2
└── index/
    ├── analyzer.py    text -> index terms
    ├── postings.py    posting lists, intersect and union merge walks
    ├── base.py        the read interface every backend satisfies
    ├── memory.py      in-memory inverted index, the correctness oracle
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
