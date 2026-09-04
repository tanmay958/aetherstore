"""An in-memory stand-in for the S3 API.

Deliberately strict about Range headers, because that is where the real bug
risk lives. HTTP ranges are inclusive at both ends, so 64 bytes from offset 0
is `bytes=0-63`. Getting that wrong returns one extra byte per read, which a
varint decoder consumes as the beginning of a value that does not exist, and
the resulting corruption looks nothing like an off-by-one.

This lets the S3 code path be tested without Docker, so CI stays offline. A
separate integration suite runs the same contract against real MinIO.
"""

from __future__ import annotations

import io

from botocore.exceptions import ClientError


def _error(code: str, operation: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, operation)


class FakeS3Client:
    """Enough of the S3 client for ObjectStore, and no more."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.range_headers: list[str] = []

    # -- the API surface S3Store uses --------------------------------------

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> dict:
        self.objects[(Bucket, Key)] = bytes(Body)
        return {}

    def get_object(self, *, Bucket: str, Key: str, Range: str | None = None) -> dict:
        try:
            data = self.objects[(Bucket, Key)]
        except KeyError:
            raise _error("NoSuchKey", "GetObject") from None

        if Range is None:
            return {"Body": io.BytesIO(data), "ContentLength": len(data)}

        self.range_headers.append(Range)
        body = self._slice(data, Range)
        return {"Body": io.BytesIO(body), "ContentLength": len(body)}

    def delete_object(self, *, Bucket: str, Key: str) -> dict:
        self.objects.pop((Bucket, Key), None)
        return {}

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        try:
            data = self.objects[(Bucket, Key)]
        except KeyError:
            raise _error("404", "HeadObject") from None
        return {"ContentLength": len(data)}

    # -- range parsing -----------------------------------------------------

    @staticmethod
    def _slice(data: bytes, header: str) -> bytes:
        if not header.startswith("bytes="):
            raise _error("InvalidRange", "GetObject")
        spec = header[len("bytes=") :]

        if spec.startswith("-"):
            # Suffix range: the last N bytes, needing no knowledge of size.
            suffix = int(spec[1:])
            if suffix == 0:
                raise _error("InvalidRange", "GetObject")
            return data[-suffix:] if suffix < len(data) else data

        first, _, last = spec.partition("-")
        start = int(first)
        if start >= len(data):
            raise _error("InvalidRange", "GetObject")
        if not last:
            return data[start:]
        # Inclusive on both ends, and clamped rather than erroring when the
        # end runs past the object, which is what S3 does.
        return data[start : int(last) + 1]
