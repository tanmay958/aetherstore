# Data

AetherStore indexes the REES46 public e-commerce clickstream.
There is no synthetic data source: the same loader reads the committed test fixture and the full multi-gigabyte file.

## Attribution

Dataset: **eCommerce behavior data from multi category store**, published by the [REES46 Marketing Platform](https://rees46.com/en/datasets).

- Kaggle: <https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store>
- Roughly 285M events across October 2019 to April 2020, from one large multi-category online store.
- Licensed as "Data files © Original Authors". Free to use with attribution to the dataset page and to REES46.

Because it is not under an open redistribution licence, **no raw rows from the dataset are committed to this repository**.
`tests/fixtures/rees46_sample.csv` is hand-authored in the REES46 schema; it is a schema fixture, not an excerpt of the data.

## Getting the data

You do not need the dataset to build or test anything.
The entire test suite runs offline against the committed fixture.
Download it when you want benchmark numbers over real volume.

1. Sign in to Kaggle and download a month from the dataset page, or use the CLI with an API token in `~/.kaggle/kaggle.json`:

   ```
   kaggle datasets download -d mkechinov/ecommerce-behavior-data-from-multi-category-store -f 2019-Oct.csv
   ```

2. Kaggle usually wraps single-file downloads in a `.zip`, so extract it once. The loader reads `.csv` and `.csv.gz`, not `.zip`.

3. Put it at `data/raw/2019-Oct.csv.gz`. Everything under `data/` is gitignored.

October is about 1.6 GB gzipped and 5.5 GB expanded.
You rarely want all of it: a few million events produce solid numbers and iterate far faster.

## Schema

As published:

| Column | Notes |
|---|---|
| `event_time` | `2019-10-01 00:00:00 UTC`, fixed width |
| `event_type` | `view`, `cart`, `remove_from_cart`, `purchase` |
| `product_id` | numeric |
| `category_id` | numeric, not indexed |
| `category_code` | dotted taxonomy, e.g. `electronics.smartphone`. **Frequently empty.** |
| `brand` | downcased. **Frequently empty.** |
| `price` | float |
| `user_id` | numeric |
| `user_session` | session uuid, rotates after a long pause |

Three things it does **not** contain, which shape the whole design:

- **No product name.** See "Derived titles" below.
- **No search events and no queries.** The `query` field is always None.
- **No device information.** The `device` field is always None.

Nothing invents a value for the last two. They stay empty.

## Derived titles

REES46's only text is `brand` and `category_code`, which is about three index terms per document.
Real catalogs carry five to ten, and the difference changes the shape of the index being measured: a smaller term dictionary, longer and less varied posting lists, and almost nothing for BM25's document-length normalization to work with.

So `aether.data.titles` derives a title per product:

```
Samsung   White     Lite      Smartphone   L486
^real     ^derived  ^derived  ^real        ^derived
```

Two rules keep this honest.

**Only the adjectives are invented.**
Brand and category leaf come straight out of the file.
A product with neither gets no title at all rather than one conjured from nothing.

**A title is a pure function of `product_id`.**
Load 100k rows today and 10M next month and product `1005105` is titled identically both times.
That rules out counters and RNG streams, and specifically rules out the builtin `hash()`, which is salted per process.
CRC32 is used instead, and a test runs the derivation in three subprocesses under different `PYTHONHASHSEED` values to prove it.

This is sound for **storage engine** measurements, which depend on the statistical shape of text rather than its truthfulness: vocabulary size, term skew, document length.
It is not sound for **relevance** claims, but REES46 ships no relevance judgments, so those were never available.

Pass `--no-derived-titles` to disable it and index only real text.

## Where data lives

| Stage | Location | Size | Committed |
|---|---|---|---|
| Unit tests, CI | `tests/fixtures/rees46_sample.csv` | ~5 KB | yes |
| Local development | `data/slices/dev.csv` via `slice.py` | ~10 MB | no |
| Benchmarks | `data/raw/2019-Oct.csv.gz` | 1.6 GB | no |
| Deployed indexing | your own private R2 bucket, read by range request | 1.6 GB | no |
| Deployed query service | nothing: it reads segments only | | no |

The last row is the important one.
The raw CSV is an input to a build step, the way source code is an input to a compiler.
The deployed search engine reads segment files from object storage and never sees a CSV, holds no Kaggle credentials, and never downloads the dataset.

## Commands

Scan and report, without writing anything:

```
uv run python -m aether.data.rees46 --input data/raw/2019-Oct.csv.gz --limit 100000
```

Convert to JSONL:

```
uv run python -m aether.data.rees46 --input data/raw/2019-Oct.csv.gz \
    --limit 1000000 --out data/events.jsonl
```

Carve a development slice of whole sessions:

```
uv run python -m aether.data.slice --input data/raw/2019-Oct.csv.gz \
    --sessions 2000 --out data/slices/dev.csv
```

The slicer keeps whole sessions rather than a row count.
A session cut in half can show an `add_to_cart` whose matching `purchase` was truncated away, which turns a converted session into a fake abandonment and poisons the label the ML phase depends on.
