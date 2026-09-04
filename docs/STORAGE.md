# Storage

Segments live in object storage.
One implementation, `S3Store`, covers AWS S3, Cloudflare R2, and MinIO, because all three speak the same protocol.
The only difference between them is an endpoint URL and a set of credentials.

This is what `ObjectStore` was defined for back in step 3, while everything was still local files.
The segment reader, the query planner, BM25, and every behavioural test are unchanged by any of this.

| URI | Goes to |
|---|---|
| `data/segments/0.seg` | a local directory |
| `file:///abs/path/0.seg` | the same, spelled out |
| `s3://aether/segments/0.seg` | MinIO or AWS S3, depending on `AETHER_S3_ENDPOINT` |
| `r2://aether/segments/0.seg` | Cloudflare R2, endpoint built from `R2_ACCOUNT_ID` |

Every command takes these interchangeably:

```
uv run python -m aether.index.build events.csv r2://aether/segments/0.seg
uv run python -m aether.index.search r2://aether/segments/0.seg "samsung smartphone"
uv run python -m aether.storage.check r2://aether
```

## Verifying a backend

Before trusting an index to a bucket, check it:

```
uv run python -m aether.storage.check r2://aether
```

It runs exactly the operations the segment reader depends on, in the order it depends on them, and times each.
A wrong credential, a missing bucket, or a backend that mishandles suffix ranges shows up in seconds rather than halfway through indexing forty million events.

The timings are worth reading.
Locally every read is a seek at around a tenth of a millisecond.
Against R2 or S3 the same reads take tens of milliseconds each.
That gap is the entire justification for the segment format, and this is the cheapest way to see it on your own connection.

## Cloudflare R2

The recommended backend, and the reason is the free tier: 10 GB of storage, 1 million writes, 10 million reads per month, **zero egress**, and no expiry.
For comparison, S3's free tier allows 20,000 GET requests per month and lasts twelve months.
A single fan-out query issues dozens of range GETs, so that budget covers a few hundred queries before it starts costing, and then it expires entirely.

Note that R2 requires a payment method on file even to stay inside the free tier.

**Setup**

1. In the Cloudflare dashboard, open **R2** and create a bucket, for example `aether`.
2. Note your **Account ID** from the R2 overview page.
3. Create an **R2 API token** with Object Read & Write on that bucket. You get an access key id and a secret access key.
4. Put them in the environment:

   ```
   export R2_ACCOUNT_ID=your_account_id
   export R2_ACCESS_KEY_ID=your_r2_access_key_id
   export R2_SECRET_ACCESS_KEY=your_r2_secret_access_key
   ```

   `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` work too, and are what most
   S3 tooling expects.

5. Check it:

   ```
   uv run python -m aether.storage.check r2://aether
   ```

### Why the credentials look like AWS credentials

No AWS account is involved, and these are not AWS credentials.
They are R2 API tokens issued by Cloudflare.

R2 has no protocol of its own: Cloudflare implemented the S3 API, including AWS Signature V4 request signing.
So from a client's point of view R2 is S3 at a different hostname, which is why boto3, aws-cli, rclone, and s3cmd all work with it unchanged.
Those libraries look for `AWS_ACCESS_KEY_ID` because that is the name the SDK defines, not because AWS is on the other end.
It is the same reason `psql` reaches Postgres, CockroachDB, and Neon through `PGHOST`: the variable names travel with the protocol, not the vendor.

The `R2_`-prefixed aliases exist purely because reading "AWS_ACCESS_KEY_ID" while configuring Cloudflare is confusing.

**Two things this code handles that would otherwise bite you**

R2 requires the signing region to be the literal string `auto` and rejects anything else.
`default_region()` returns it whenever the endpoint is an R2 host.

boto3 1.36 began attaching CRC32 checksums to uploads by default, and R2 rejects the header with `Header 'x-amz-checksum-algorithm' with value 'CRC32' not implemented`.
The client sets `request_checksum_calculation="when_required"` whenever a custom endpoint is configured, restoring the older behaviour.
Real AWS keeps its defaults, and therefore its upload integrity checks, because that branch does not run for it.

## MinIO, locally

```
make up
```

Starts MinIO on `localhost:9000` with a console on `localhost:9001`, and creates the `aether` bucket.
Credentials are `aether` / `aethersecret`.

```
export AETHER_S3_ENDPOINT=http://localhost:9000
export AWS_ACCESS_KEY_ID=aether
export AWS_SECRET_ACCESS_KEY=aethersecret

uv run python -m aether.storage.check s3://aether
uv run pytest -m integration
```

MinIO is the first container in this project, and it arrives at step 7 rather than step 0 deliberately.
The segment format, the codec, the byte-range reader, and BM25 needed no infrastructure at all.
Building them first meant no time was lost to container configuration before there was anything to run in one.

## AWS S3

```
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1

uv run python -m aether.storage.check s3://your-bucket
```

Leave `AETHER_S3_ENDPOINT` unset.
Co-locating compute with the bucket matters here: a range read within one region is single-digit milliseconds, and across the internet it is tens.

If credential resolution fails with `Missing Dependency ... botocore[crt]`, an SSO or login profile is being picked up.
Either set `AWS_PROFILE` to a key-based profile or install the extra.

## The cost model, restated

| Operation | Latency | Cost |
|---|---|---|
| RAM read | ~100 ns | free |
| Local seek | ~100 us | free |
| S3 or R2 range GET | 20-50 ms | billed per request, same for 64 B or 8 MB |

The number of requests is what you pay in both latency and money, and it does not change when you move from a laptop to the cloud.
That is why `CountingStore` reports requests rather than only milliseconds, and why every command prints them:

```
  storage cost
    open (once)      2 requests, 290 B
    term lookup      1 request, 194 B
    posting lists    2 requests, 28 B
    fetch  4 docs     1 request, 966 B
```
