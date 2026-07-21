from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from mimeme import storage

_READ_CHUNK = 1024 * 1024
_PUT_BYTES_MAX = 5 * 1024 * 1024


@dataclass
class _Entry:
    data: bytes
    content_type: str
    checksum: storage.Checksum
    modified_at: datetime


class Memory:
    """In-memory `storage.Store` for tests and Env-level fakes."""

    def __init__(self, *, put_bytes_max: int = _PUT_BYTES_MAX) -> None:
        self._objects: dict[str, _Entry] = {}
        self._put_bytes_max = put_bytes_max
        self.closed = False

    async def put(
        self,
        obj: storage.Object,
        body: AsyncIterable[bytes],
        *,
        length: int,
        content_type: str,
        checksum: storage.Checksum,
    ) -> storage.Info:
        buffer = bytearray()
        async for chunk in body:
            buffer += chunk
        data = bytes(buffer)
        if len(data) != length:
            raise storage.Integrity(f"declared length {length} != {len(data)} for {obj.key}")
        if storage.Checksum.of(data).value != checksum.value:
            raise storage.Integrity(f"checksum mismatch for {obj.key}")
        return self._store(obj, data, content_type=content_type, checksum=checksum)

    async def put_bytes(
        self, obj: storage.Object, data: bytes, *, content_type: str
    ) -> storage.Info:
        if len(data) > self._put_bytes_max:
            raise storage.Invalid(
                f"put_bytes limit is {self._put_bytes_max} bytes, got {len(data)}"
            )
        return self._store(obj, data, content_type=content_type, checksum=storage.Checksum.of(data))

    def _store(
        self, obj: storage.Object, data: bytes, *, content_type: str, checksum: storage.Checksum
    ) -> storage.Info:
        self._objects[obj.key] = _Entry(
            data=data,
            content_type=content_type,
            checksum=checksum,
            modified_at=datetime.now(UTC),
        )
        return self._info(obj, self._objects[obj.key])

    def read(self, obj: storage.Object) -> AbstractAsyncContextManager[AsyncIterator[bytes]]:
        return self._read(obj)

    @asynccontextmanager
    async def _read(self, obj: storage.Object) -> AsyncIterator[AsyncIterator[bytes]]:
        entry = self._objects.get(obj.key)
        if entry is None:
            raise storage.Missing(obj.key)
        yield _stream(entry.data)

    async def read_bytes(self, obj: storage.Object, *, max_bytes: int) -> bytes:
        entry = self._objects.get(obj.key)
        if entry is None:
            raise storage.Missing(obj.key)
        if len(entry.data) > max_bytes:
            raise storage.Invalid(f"object {obj.key} exceeds max_bytes={max_bytes}")
        return entry.data

    async def stat(self, obj: storage.Object) -> storage.Info | None:
        entry = self._objects.get(obj.key)
        return self._info(obj, entry) if entry else None

    async def delete(self, obj: storage.Object) -> None:
        self._objects.pop(obj.key, None)

    async def list(self, *, prefix: str = "") -> AsyncIterator[storage.Info]:
        for key in sorted(self._objects):
            if key.startswith(prefix):
                yield self._info(storage.Object(key), self._objects[key])

    async def probe(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True

    def _info(self, obj: storage.Object, entry: _Entry) -> storage.Info:
        return storage.Info(
            object=obj,
            length=len(entry.data),
            content_type=entry.content_type,
            checksum=entry.checksum,
            etag=entry.checksum.value,
            modified_at=entry.modified_at,
        )


async def _stream(data: bytes) -> AsyncIterator[bytes]:
    for start in range(0, len(data), _READ_CHUNK):
        yield data[start : start + _READ_CHUNK]
