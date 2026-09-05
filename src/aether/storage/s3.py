"""An object store backed by the S3 API.

One implementation covers AWS S3, Cloudflare R2, and MinIO, because all three
speak the same protocol. The only thing that differs between them is the
endpoint URL and the credentials, which is the entire reason step 3 defined
`ObjectStore` while everything was still local: the segment reader, the query
planner, and every test are unchanged by this file existing.

The method that matters is `get_range`, and its subtlety is that HTTP ranges
are inclusive at both ends. Asking for 64 bytes starting at 0 means
`bytes=0-63`, not `bytes=0-64`. An off-by-one there silently returns one extra
byte per read, which a varint decoder will happily consume as the start of a
value that does not exist.

Cost changes character here in a way the interface deliberately hides. A range
read against LocalStore is a seek: about 100 microseconds, and free. The same
call against S3 is a 20-50ms round trip that is billed per request regardless
of whether it returns 64 bytes or 8 megabytes. Nothing above this layer needs
to change, but the request counts that `CountingStore` has been reporting all
along stop being an abstract number and start being the latency and the bill.

Two things about non-AWS backends are handled here rather than left as
surprises during a deploy. R2 requires the region to be the literal string
"auto". And boto3 1.36 began attaching CRC32 checksums to uploads by default,
which R2 and several other S3-compatible stores reject outright with "Header
'x-amz-checksum-algorithm' with value 'CRC32' not implemented"; restoring the
older behaviour fixes it. Both are applied only when a custom endpoint is set,
so real AWS keeps its defaults and its upload integrity checks.
"""

from __future__ import annotations

import os
from typing import Any, Iterator

from aether.storage.base import ObjectStore

# Error codes S3 and its clones use for "that object is not there". They are
# not consistent between implementations, hence the set.
_NOT_FOUND = frozenset({"404", "NoSuchKey", "NoSuchBucket", "NotFound"})

# Returned when a range starts past the end of an object. LocalStore returns
# empty bytes in that situation, so this one does too.
_RANGE_NOT_SATISFIABLE = frozenset({"416", "InvalidRange", "RequestedRangeNotSatisfiable"})


# Cloudflare R2's endpoint host. Used only to pick the right default region.
R2_HOST_SUFFIX = "r2.cloudflarestorage.com"


def default_region(endpoint_url: str | None) -> str | None:
    """The region to sign with, given an endpoint.

    R2 demands the literal string "auto" and rejects anything else. Other
    S3-compatible stores such as MinIO ignore the region but still need one
    present for SigV4 to produce a matching signature. Real AWS is left to
    resolve its own region from the environment or config file.
    """
    if not endpoint_url:
        return None
    if R2_HOST_SUFFIX in endpoint_url:
        return "auto"
    return "us-east-1"


def _build_client(
    endpoint_url: str | None,
    region: str | None,
    pool: int,
    access_key: str | None = None,
    secret_key: str | None = None,
) -> Any:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "S3Store needs boto3. Install it with:  uv sync --extra s3"
        ) from exc

    settings: dict[str, Any] = {
        # The query planner fans out across segments concurrently, so the pool
        # has to be wide enough that connection reuse does not become the
        # bottleneck instead of the network.
        "max_pool_connections": pool,
        "retries": {"max_attempts": 3, "mode": "standard"},
    }

    if endpoint_url:
        # boto3 1.36 started attaching CRC32 checksums to uploads by default.
        # R2 rejects the header outright, and other S3 clones have had the
        # same trouble, so restore the previous behaviour for anything that is
        # not AWS. Real S3 keeps its defaults, and therefore its integrity
        # checks, because this branch does not run for it.
        settings["request_checksum_calculation"] = "when_required"
        settings["response_checksum_validation"] = "when_required"
        settings["signature_version"] = "s3v4"

    # A dedicated session per store, rather than the module-level
    # boto3.client(). That helper resolves credentials once into a
    # process-wide default session and reuses them, so a process holding two
    # stores against different backends, say MinIO and R2, would silently sign
    # both with whichever credentials were resolved first. A per-store session
    # keeps them independent, and makes rotation take effect.
    return boto3.session.Session().client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region,
        # Left as None unless explicitly supplied, so boto3's normal
        # resolution order still applies: environment, shared config file,
        # instance role. Passing keys here is only for callers that read them
        # from somewhere else, such as R2_ACCESS_KEY_ID.
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(**settings),
    )


class S3Store(ObjectStore):
    """Objects in an S3-compatible bucket, optionally under a key prefix.

        # MinIO, locally
        S3Store("aether", endpoint_url="http://localhost:9000")

        # Cloudflare R2
        S3Store("aether", endpoint_url="https://<account>.r2.cloudflarestorage.com")

        # AWS S3
        S3Store("aether", region="us-east-1")
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        endpoint_url: str | None = None,
        region: str | None = None,
        client: Any | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        max_pool_connections: int = 64,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.endpoint_url = endpoint_url or os.getenv("AETHER_S3_ENDPOINT")
        self.region = region or os.getenv("AWS_REGION") or default_region(self.endpoint_url)
        self._client = client or _build_client(
            self.endpoint_url, self.region, max_pool_connections, access_key, secret_key
        )

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def _get(self, key: str, byte_range: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            response = self._client.get_object(
                Bucket=self.bucket, Key=self._key(key), Range=byte_range
            )
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in _RANGE_NOT_SATISFIABLE:
                return b""
            if code in _NOT_FOUND:
                raise FileNotFoundError(f"{self.bucket}/{self._key(key)}") from exc
            raise
        return response["Body"].read()

    def get_range(self, key: str, start: int, length: int) -> bytes:
        if length <= 0:
            return b""
        # Inclusive at both ends, so the last byte wanted is start + length - 1.
        return self._get(key, f"bytes={start}-{start + length - 1}")

    def get_suffix(self, key: str, length: int) -> bytes:
        if length <= 0:
            return b""
        # A suffix range needs no knowledge of the object's size, which is how
        # a reader finds a segment's footer in a file it has never seen.
        # Asking for more than exists returns the whole object rather than an
        # error, matching LocalStore.
        return self._get(key, f"bytes=-{length}")

    def put(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def delete(self, key: str) -> None:
        # S3 treats deleting a missing key as success, which is the behaviour
        # a retrying caller needs.
        self._client.delete_object(Bucket=self.bucket, Key=self._key(key))

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        # Paginated, because a bucket can hold more objects than one response
        # carries and a truncated listing would make the collector think
        # everything past the first page is unreferenced.
        paginator = self._client.get_paginator("list_objects_v2")
        full_prefix = self._key(prefix) if prefix else self.prefix
        strip = len(self.prefix) + 1 if self.prefix else 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for entry in page.get("Contents", []):
                yield entry["Key"][strip:]

    def _head(self, key: str) -> dict:
        from botocore.exceptions import ClientError

        try:
            return self._client.head_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in _NOT_FOUND:
                raise FileNotFoundError(f"{self.bucket}/{self._key(key)}") from exc
            raise

    def modified_at(self, key: str) -> float:
        return self._head(key)["LastModified"].timestamp()

    def size(self, key: str) -> int:
        return self._head(key)["ContentLength"]

    def exists(self, key: str) -> bool:
        try:
            self.size(key)
        except FileNotFoundError:
            return False
        return True
