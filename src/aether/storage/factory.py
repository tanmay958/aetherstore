"""Pick a store from a URI.

This is the "one config line" the ObjectStore abstraction was built for. The
same segment, the same reader, and the same tests work against a directory, a
MinIO container, an R2 bucket, or S3, and the only thing that changes is this
string.

    data/segments                    a local directory
    file:///abs/path                 the same, spelled out
    s3://aether                      a bucket
    s3://aether/indexes/clickstream  a bucket with a key prefix
    r2://aether                      Cloudflare R2, endpoint built for you

Credentials and endpoint come from the environment, the way every AWS tool
expects:

    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AETHER_S3_ENDPOINT      http://localhost:9000 for MinIO,
                            unset for real AWS
    R2_ACCOUNT_ID           for r2://, which builds the endpoint from it
    AWS_REGION              optional; "auto" is forced for R2

See docs/STORAGE.md for the R2 and MinIO setup.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from aether.storage.base import ObjectStore
from aether.storage.local import LocalStore

REMOTE_SCHEMES = ("s3", "r2")


def r2_endpoint() -> str:
    """R2's account-scoped endpoint, from the environment.

    An explicit AETHER_S3_ENDPOINT wins, so a proxy or a custom domain can be
    pointed at without changing any URI.
    """
    explicit = os.getenv("AETHER_S3_ENDPOINT")
    if explicit:
        return explicit
    account = os.getenv("R2_ACCOUNT_ID")
    if not account:
        raise ValueError(
            "r2:// needs R2_ACCOUNT_ID (or AETHER_S3_ENDPOINT) in the environment. "
            "See docs/STORAGE.md."
        )
    return f"https://{account}.r2.cloudflarestorage.com"


def open_store(uri: str | Path) -> ObjectStore:
    """Build the store a URI names, treating any trailing path as a prefix."""
    text = str(uri)
    parsed = urlparse(text)

    if parsed.scheme in REMOTE_SCHEMES:
        from aether.storage.s3 import S3Store

        if not parsed.netloc:
            raise ValueError(f"{text!r} names no bucket; expected s3://bucket/prefix")
        endpoint = r2_endpoint() if parsed.scheme == "r2" else None
        return S3Store(parsed.netloc, prefix=parsed.path.lstrip("/"), endpoint_url=endpoint)

    if parsed.scheme == "file":
        return LocalStore(parsed.path)

    # A bare path, including Windows drive letters, which urlparse reads as a
    # single-character scheme.
    if not parsed.scheme or len(parsed.scheme) == 1:
        return LocalStore(text)

    raise ValueError(
        f"unsupported storage scheme {parsed.scheme!r} in {text!r}; "
        "expected a path, file://, or s3://"
    )


def open_object(uri: str | Path) -> tuple[ObjectStore, str]:
    """Split a URI naming one object into the store holding it and its key.

        s3://aether/segments/p0/0.seg  ->  S3Store("aether", prefix="segments/p0"), "0.seg"
        data/segments/0.seg            ->  LocalStore("data/segments"), "0.seg"

    Keeping the directory in the store and only the filename as the key means
    a reader is handed the smallest scope it needs, which is what the
    segment-per-file design assumes.
    """
    text = str(uri)
    parsed = urlparse(text)

    if parsed.scheme in REMOTE_SCHEMES or parsed.scheme == "file":
        path = PurePosixPath(parsed.path.lstrip("/"))
        if not path.name:
            raise ValueError(f"{text!r} names no object")
        parent = str(path.parent) if str(path.parent) != "." else ""
        if parsed.scheme == "file":
            return LocalStore("/" + parent if parent else "/"), path.name
        return open_store(f"{parsed.scheme}://{parsed.netloc}/{parent}"), path.name

    local = Path(text)
    return LocalStore(local.parent), local.name
