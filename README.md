# AetherStore

A distributed search and streaming prediction engine, built from scratch.

An inverted index with its own on-disk segment format and compression, queried over object storage using HTTP range requests, plus a horizontally partitioned streaming inference pipeline over the same event stream.

Not a wrapper around Elasticsearch. The segment format, the postings codec, the term dictionary, the query planner, and the cost accounting are the project.

## Status

**Live:** https://aether-441173057461.us-central1.run.app

Searching 100,000 real REES46 events on Cloudflare R2, and scoring sessions
with a model trained on 10 million of them. Scales to zero, and costs nothing.

```
curl "https://aether-441173057461.us-central1.run.app/api/search?q=samsung+smartphone&k=3&explain=true"
```

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
| 5.0 | Cart-abandonment model, streaming inference | done |
| 6.0 | HTTP query service, containerised | done |
| 6.1 | Numpy-only model export, no pickle at serving | done |
| 8.0 | Deployed to Cloud Run, free tier | done |
| 6.2 | Query cost caps, shared-secret gate | done |
| 7.0 | Dashboard on Cloudflare Pages | done |

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
query by **10x in requests**, and answers are identical -- same hits, same
document ids, same BM25 scores to nine decimal places, verified at a million
documents.

### What compaction does not fix

Run against real R2, requests fell 124 to 15 and wall time only fell 2,373 ms
to 2,000 ms. Tracing every read explains why:

```
 532 ms  manifest
 770 ms  footers        ─┐  the two segments run
1118 ms  hotcaches       │  in parallel at every
1337 ms  dict block      │  level, so width is
1573 ms  dict block      │  already free
1782 ms  postings        │
1973 ms  postings       ─┘
```

**Latency is set by the depth of the dependency chain, not the number of
requests.** A dictionary block cannot be read until the hotcache is, and the
hotcache cannot be located until the footer is. Compaction reduces the *width*
of that fan-out, which was never the bottleneck once there were enough workers.

Bytes moved barely change either. Total hotcache went 5.0 MB to 4.5 MB, because
compaction consolidates it rather than shrinking it -- per segment it grew from
50 KB to 450 KB, since the hotcache carries a 32-bit length per document.

So compaction is a **cost** win, not a latency win, and on object storage
requests are billed. The three things that would cut latency, now measured
rather than guessed:

- **coalescing** adjacent range reads, which would collapse two dictionary
  reads and two postings reads into one each: depth 7 to 5
- **one byte per document length** instead of four, the way Lucene stores
  norms: hotcache 4.5 MB to about 1.1 MB
- keeping footers and hotcaches across invocations, which already happens
  within a process and is why a warm query costs a fraction of a cold one

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

## The service

One process serves both halves of the project, because both are stateless
reads of an artifact in a bucket and the same dashboard wants both.

```
uv run python -m aether.service --index r2://aether/idx100k --model r2://aether/models/model.pkl
```

| | |
|---|---|
| `GET /health` | index and model, and which of them is degraded |
| `GET /api/search?q=&k=&mode=&global_stats=&explain=` | ranked documents |
| `POST /api/predict` | P(abandon) after each event of a session |
| `GET /api/index`, `GET /api/model` | manifest summary, model card |

Against the 100,000-document index on R2:

```
cold    246 ms   10 requests, 2.9 KB
warm    0.5 ms    0 requests
```

The warm figure is not a cache of answers. A segment's footer and hotcache are
read once and held for the life of the process, and they can never go stale
because a segment is immutable. Publishing new data writes a new segment and a
new manifest, so nothing has to be invalidated.

`?explain=1` reports exactly what a query spent. It takes a lock, because the
read counters are shared and concurrent requests would otherwise land in each
other's totals, so it is opt-in and off the default path. Measurement should
not shape the code it measures.

### Prediction over HTTP

`/api/predict` returns the whole trajectory rather than one number, because
what is interesting about this model is how the probability moves:

```
0  view                              P(abandon) = -
1  view                              P(abandon) = -
2  add_to_cart   cart $1206.45       P(abandon) = 0.5354
3  view          +310s idle          P(abandon) = 0.7479
4  view          +800s idle          P(abandon) = 0.8494
```

The endpoint drives `Predictor.handle`, the same function the Kafka replicas
run, rather than reimplementing scoring for HTTP. A test asserts the two paths
produce identical probabilities, because a second implementation here is
exactly how train/serve skew gets in.

### Cold start

A scale-to-zero service pays this on the first request after every idle
period, so it is the number worth attacking. Measured in the container,
against R2:

```
                              before    after
import scikit-learn            783 ms       -     dropped
numpy, boto3, fastapi          496 ms   658 ms
load index and model from R2 1,433 ms   614 ms    concurrent, smaller model
                             --------  --------
                             2,718 ms 1,273 ms
image (uncompressed)           693 MB   412 MB
```

Two changes, and neither was a micro-optimisation.

**The model no longer needs scikit-learn.** A fitted gradient boosting
ensemble is a pile of thresholds; `aether.ml.export` writes it out as flat
numpy arrays with a JSON header. Serving loads that instead of the pickle:
783 ms and most of the image, for code that runs during training and never
again. The exported scorer agrees with scikit-learn to 1.7e-16, and is 98
times faster on the single-row scoring that serving actually does, 0.090 ms
against 8.783 ms, because it skips the array overhead for one row.

The stronger reason is not speed. **Unpickling a model is arbitrary code
execution**, so loading a `.pkl` from a bucket means whoever can write to that
bucket runs code in the serving container. The `.npz` is read with
`allow_pickle=False`: it is data, and cannot execute.

**The index and the model load at once.** They share nothing, and a cold start
is almost all waiting rather than working, so doing them in sequence added a
full round trip. The same point compaction made: latency here is bound by the
depth of the chain of waits, not by how much is read.

What remains is 614 ms of object storage round trips and 658 ms of importing
numpy, boto3 and fastapi. Both are close to the floor for this shape.

The image deliberately excludes the Kafka client too. A container that scales
to zero cannot be a consumer, so shipping librdkafka would pay tens of
megabytes for something that can never run there.

### Deployed

Measured on Cloud Run, reading R2 from `us-central1`:

```
cold start, first request after idle     514 ms
warm query, terms already cached         297 ms   10 requests
warm query, new terms                    618 ms   21 requests
warm query, many matching segments     1,266 ms   40 requests
```

A warm instance still spends requests, and it should. Only a segment's footer
and hotcache are cached, and those are what make it openable; a query for a
term nobody has asked for still reads that term's dictionary block and
postings. Caching removes the cost of *opening*, paid once, not the cost of
*reading*, paid per distinct term.

The first request against a cold instance measured 3,453 ms for the same
query. That is ten segments being opened, and it is the compaction argument
restated.

Deployment, cost, and the reasoning about what is deliberately not hosted:
[docs/DEPLOY.md](docs/DEPLOY.md).

### Two model formats

```
model.pkl   723 KB   scikit-learn, for retraining and explaining
model.npz   137 KB   numpy only, for serving
```

`aether.ml.train` writes both. Serving should always be pointed at the `.npz`;
`/api/model` reports which one a container actually loaded, so an operator can
tell.

## Layout

```
src/aether/
├── events.py          canonical event schema, the boundary everything targets
├── ml/
│   ├── export.py      fitted trees -> flat numpy arrays, no pickle
│   └── loader.py      picks the format by suffix
├── service/
│   ├── app.py         FastAPI: /api/search, /api/predict, /health
│   ├── state.py       index and model, loaded once per process
│   └── main.py        uvicorn entry point, PORT-aware for Cloud Run
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

## The dashboard

`web/` is a static page and one Cloudflare Pages Function. No framework, no
build step: what is written is what is served.

The search panel shows what each query cost in storage requests and bytes,
which is the only way the engine underneath is visible at all. A results list
looks the same whether it came from this or from a library.

The prediction panel builds a session event by event and scores it against the
deployed model. Time only advances when you press "wait", because idle time is
one of the features and letting wall clock drive it would make the demo
unreproducible.

```
python3 web/dev-server.py      # serves web/ and proxies /api/* to Cloud Run
```

That proxy is not a convenience. In production the page and the API share an
origin, which is what lets the API key live in the Pages Function rather than
in the browser. Opening `index.html` straight against Cloud Run is two
origins, so the browser asks for CORS and is refused. Adding CORS to the
service to make local development easier would weaken the same-origin property
the deployment relies on, to fix a problem that only exists on a laptop.

### Why the key is in the Function and not the page

A browser cannot keep a secret: anything the page holds is in the network tab.
CORS does not help either, since browsers enforce it and `curl` ignores it. So
the secret lives at the edge, where the visitor cannot read it, and the page
calls its own origin.

`AETHER_API_KEY` is unset by default, which leaves the API open. That is the
honest default for a demo over a public dataset, and it is why the query caps
matter more than the gate: what needed defending was the bill.
