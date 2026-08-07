from __future__ import annotations

from typing import Protocol, runtime_checkable

from mimeme.search.model import Batch, Load, Loaded, Query, Status


@runtime_checkable
class Client(Protocol):
    async def query(self, query: Query, *, count: int, cursor: str | None = None) -> Batch: ...

    async def status(self) -> Status: ...

    async def close(self) -> None: ...


@runtime_checkable
class Activation(Protocol):
    async def load(self, generation: Load) -> Loaded: ...

    async def switch(self, version: str) -> Status: ...

    async def rollback(self, failed_version: str) -> Status: ...

    async def clear(self) -> Status: ...

    async def status(self) -> Status: ...
