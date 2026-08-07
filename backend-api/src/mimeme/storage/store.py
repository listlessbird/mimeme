from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Protocol, runtime_checkable

from mimeme.storage.model import Checksum, Info, Object


@runtime_checkable
class Store(Protocol):
    async def put(
        self,
        obj: Object,
        body: AsyncIterable[bytes],
        *,
        length: int,
        content_type: str,
        checksum: Checksum,
    ) -> Info: ...

    async def put_bytes(self, obj: Object, data: bytes, *, content_type: str) -> Info: ...

    def read(self, obj: Object) -> AbstractAsyncContextManager[AsyncIterator[bytes]]: ...

    async def read_bytes(self, obj: Object, *, max_bytes: int) -> bytes: ...

    async def stat(self, obj: Object) -> Info | None: ...

    async def delete(self, obj: Object) -> None: ...

    def list(self, *, prefix: str = "") -> AsyncIterator[Info]: ...

    async def probe(self) -> None: ...

    async def close(self) -> None: ...
