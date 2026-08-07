from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from mimeme.index.model import Build, Result, Seal, SealResult

Progress = Callable[[str, float], Awaitable[None]]


@runtime_checkable
class Client(Protocol):
    async def build(self, request: Build, *, progress: Progress | None = None) -> Result: ...

    async def seal(self, request: Seal) -> SealResult: ...

    async def close(self) -> None: ...
