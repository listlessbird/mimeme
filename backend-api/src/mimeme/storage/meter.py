from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from math import ceil

from mimeme.storage.model import Checksum, Counts, Info, Object
from mimeme.storage.model import Config as StorageConfig
from mimeme.storage.store import Store

_LIST_PAGE = 1000


class Meter:
    def __init__(
        self,
        store: Store,
        *,
        multipart_threshold: int = StorageConfig.model_fields["multipart_threshold"].default,
        multipart_chunk: int = StorageConfig.model_fields["multipart_chunk"].default,
    ) -> None:
        self._store = store
        self._multipart_threshold = multipart_threshold
        self._multipart_chunk = multipart_chunk
        self._counts: dict[str, int] = {}

    @property
    def counts(self) -> Counts:
        return Counts(**self._counts)

    def _record(self, operation: str, times: int = 1) -> None:
        self._counts[operation] = self._counts.get(operation, 0) + times

    async def put(
        self,
        obj: Object,
        body: AsyncIterable[bytes],
        *,
        length: int,
        content_type: str,
        checksum: Checksum,
    ) -> Info:
        if length <= self._multipart_threshold:
            self._record("put_object")
        else:
            self._record("create_multipart")
            self._record("upload_part", ceil(length / self._multipart_chunk))
            self._record("complete_multipart")
        return await self._store.put(
            obj, body, length=length, content_type=content_type, checksum=checksum
        )

    async def put_bytes(self, obj: Object, data: bytes, *, content_type: str) -> Info:
        self._record("put_object")
        return await self._store.put_bytes(obj, data, content_type=content_type)

    def read(self, obj: Object) -> AbstractAsyncContextManager[AsyncIterator[bytes]]:
        return self._read(obj)

    @asynccontextmanager
    async def _read(self, obj: Object) -> AsyncIterator[AsyncIterator[bytes]]:
        self._record("get_object")
        async with self._store.read(obj) as chunks:
            yield chunks

    async def read_bytes(self, obj: Object, *, max_bytes: int) -> bytes:
        self._record("get_object")
        return await self._store.read_bytes(obj, max_bytes=max_bytes)

    async def stat(self, obj: Object) -> Info | None:
        self._record("head_object")
        return await self._store.stat(obj)

    async def delete(self, obj: Object) -> None:
        await self._store.delete(obj)

    async def list(self, *, prefix: str = "") -> AsyncIterator[Info]:
        seen = 0
        self._record("list_page")
        async for info in self._store.list(prefix=prefix):
            seen += 1
            if seen % _LIST_PAGE == 0:
                self._record("list_page")
            yield info

    async def probe(self) -> None:
        self._record("head_bucket")
        await self._store.probe()

    async def close(self) -> None:
        await self._store.close()
