from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any, Self

from aiobotocore.config import AioConfig
from aiobotocore.session import get_session
from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    HTTPClientError,
    ReadTimeoutError,
)
from botocore.exceptions import (
    ConnectionError as BotoConnectionError,
)

from mimeme.storage.model import (
    Checksum,
    Config,
    Denied,
    Error,
    Info,
    Integrity,
    Invalid,
    Missing,
    Object,
    Timeout,
    Unavailable,
)

if TYPE_CHECKING:
    from types_aiobotocore_s3 import S3Client
    from types_aiobotocore_s3.type_defs import CompletedPartTypeDef

_READ_CHUNK = 1024 * 1024


def _map_error(exc: BaseException) -> Error:
    if isinstance(exc, Error):
        return exc
    if isinstance(exc, (ConnectTimeoutError, ReadTimeoutError, asyncio.TimeoutError)):
        return Timeout(str(exc))
    if isinstance(exc, (EndpointConnectionError, BotoConnectionError, HTTPClientError)):
        return Unavailable(str(exc))
    if isinstance(exc, ClientError):
        error = exc.response.get("Error", {})
        code = str(error.get("Code", ""))
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"NoSuchKey", "NoSuchBucket", "NotFound", "404"} or status == 404:
            return Missing(code or "not found")
        if (
            code in {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch", "403"}
            or status == 403
        ):
            return Denied(code or "access denied")
        if code in {"BadDigest", "InvalidDigest", "XAmzContentSHA256Mismatch"}:
            return Integrity(code)
        if status == 400 or code in {"InvalidRequest", "InvalidArgument", "EntityTooSmall"}:
            return Invalid(code or "invalid request")
        if isinstance(status, int) and status >= 500:
            return Unavailable(code or f"http {status}")
        return Unavailable(code or "s3 error")
    return Unavailable(str(exc))


async def _rechunk(body: AsyncIterable[bytes], size: int) -> AsyncIterator[bytes]:
    buffer = bytearray()
    async for chunk in body:
        buffer += chunk
        while len(buffer) >= size:
            yield bytes(buffer[:size])
            del buffer[:size]
    if buffer:
        yield bytes(buffer)


class S3:
    def __init__(self, config: Config, client: S3Client, ctx: Any) -> None:
        self._config = config
        self._client = client
        self._ctx = ctx

    @classmethod
    async def open(cls, config: Config) -> Self:
        session = get_session()
        ctx = session.create_client(
            "s3",
            region_name=config.region,
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key.get_secret_value(),
            config=AioConfig(
                s3={"addressing_style": "path" if config.force_path_style else "auto"},
                signature_version="s3v4",
            ),
        )
        client = await ctx.__aenter__()
        return cls(config, client, ctx)

    @property
    def bucket(self) -> str:
        return self._config.bucket

    async def put(
        self,
        obj: Object,
        body: AsyncIterable[bytes],
        *,
        length: int,
        content_type: str,
        checksum: Checksum,
    ) -> Info:
        if length <= self._config.multipart_threshold:
            return await self._put_single(obj, body, content_type=content_type, checksum=checksum)
        return await self._put_multipart(obj, body, content_type=content_type, checksum=checksum)

    async def put_bytes(self, obj: Object, data: bytes, *, content_type: str) -> Info:
        if len(data) > self._config.put_bytes_max:
            raise Invalid(f"put_bytes limit is {self._config.put_bytes_max} bytes, got {len(data)}")
        checksum = Checksum.of(data)
        return await self._put_single(
            obj, _once(data), content_type=content_type, checksum=checksum, known=data
        )

    async def _put_single(
        self,
        obj: Object,
        body: AsyncIterable[bytes],
        *,
        content_type: str,
        checksum: Checksum,
        known: bytes | None = None,
    ) -> Info:
        if known is not None:
            data = known
        else:
            buffer = bytearray()
            async for chunk in body:
                buffer += chunk
            data = bytes(buffer)
        actual = Checksum.of(data)
        if actual.value != checksum.value:
            raise Integrity(f"checksum mismatch for {obj.key}")
        try:
            resp = await self._client.put_object(
                Bucket=self.bucket,
                Key=obj.key,
                Body=data,
                ContentType=content_type,
                Metadata={"sha256": checksum.value},
            )
        except Exception as exc:
            raise _map_error(exc) from exc
        return Info(
            object=obj,
            length=len(data),
            content_type=content_type,
            checksum=checksum,
            etag=_etag(resp.get("ETag")),
        )

    async def _put_multipart(
        self,
        obj: Object,
        body: AsyncIterable[bytes],
        *,
        content_type: str,
        checksum: Checksum,
    ) -> Info:
        try:
            created = await self._client.create_multipart_upload(
                Bucket=self.bucket,
                Key=obj.key,
                ContentType=content_type,
                Metadata={"sha256": checksum.value},
            )
        except Exception as exc:
            raise _map_error(exc) from exc

        upload_id = created["UploadId"]
        digest = hashlib.sha256()
        total = 0
        parts: list[CompletedPartTypeDef] = []
        try:
            number = 1
            async for part in _rechunk(body, self._config.multipart_chunk):
                digest.update(part)
                total += len(part)
                resp = await self._client.upload_part(
                    Bucket=self.bucket,
                    Key=obj.key,
                    PartNumber=number,
                    UploadId=upload_id,
                    Body=part,
                )
                parts.append({"ETag": resp["ETag"], "PartNumber": number})
                number += 1
            if digest.hexdigest() != checksum.value:
                raise Integrity(f"checksum mismatch for {obj.key}")
            completed = await self._client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=obj.key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )
        except Exception as exc:
            with suppress(Exception):
                await self._client.abort_multipart_upload(
                    Bucket=self.bucket, Key=obj.key, UploadId=upload_id
                )
            raise _map_error(exc) from exc

        return Info(
            object=obj,
            length=total,
            content_type=content_type,
            checksum=checksum,
            etag=_etag(completed.get("ETag")),
        )

    def read(self, obj: Object) -> AbstractAsyncContextManager[AsyncIterator[bytes]]:
        return self._read(obj)

    @asynccontextmanager
    async def _read(self, obj: Object) -> AsyncIterator[AsyncIterator[bytes]]:
        try:
            resp = await self._client.get_object(Bucket=self.bucket, Key=obj.key)
        except Exception as exc:
            raise _map_error(exc) from exc
        body = resp["Body"]
        async with body:
            yield body.iter_chunks(_READ_CHUNK)

    async def read_bytes(self, obj: Object, *, max_bytes: int) -> bytes:
        buffer = bytearray()
        async with self._read(obj) as chunks:
            async for chunk in chunks:
                buffer += chunk
                if len(buffer) > max_bytes:
                    raise Invalid(f"object {obj.key} exceeds max_bytes={max_bytes}")
        return bytes(buffer)

    async def stat(self, obj: Object) -> Info | None:
        try:
            resp = await self._client.head_object(Bucket=self.bucket, Key=obj.key)
        except Exception as exc:
            mapped = _map_error(exc)
            if isinstance(mapped, Missing):
                return None
            raise mapped from exc
        return self._info_from_head(obj, resp)

    async def delete(self, obj: Object) -> None:
        try:
            await self._client.delete_object(Bucket=self.bucket, Key=obj.key)
        except Exception as exc:
            mapped = _map_error(exc)
            if isinstance(mapped, Missing):
                return
            raise mapped from exc

    async def list(self, *, prefix: str = "") -> AsyncIterator[Info]:
        paginator = self._client.get_paginator("list_objects_v2")
        try:
            async for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for item in page.get("Contents", []):
                    key = item.get("Key")
                    if key is None:
                        continue
                    yield Info(
                        object=Object(key),
                        length=int(item.get("Size", 0)),
                        etag=_etag(item.get("ETag")),
                        modified_at=item.get("LastModified"),
                    )
        except Exception as exc:
            raise _map_error(exc) from exc

    async def probe(self) -> None:
        try:
            await self._client.head_bucket(Bucket=self.bucket)
        except Exception as exc:
            raise _map_error(exc) from exc

    async def close(self) -> None:
        await self._ctx.__aexit__(None, None, None)

    def _info_from_head(self, obj: Object, resp: Any) -> Info:
        metadata = resp.get("Metadata", {})
        recorded = metadata.get("sha256")
        checksum = Checksum(value=recorded) if recorded else None
        return Info(
            object=obj,
            length=int(resp.get("ContentLength", 0)),
            content_type=resp.get("ContentType"),
            checksum=checksum,
            etag=_etag(resp.get("ETag")),
            modified_at=resp.get("LastModified"),
        )


async def _once(data: bytes) -> AsyncIterator[bytes]:
    yield data


def _etag(value: str | None) -> str | None:
    return value.strip('"') if value else None
