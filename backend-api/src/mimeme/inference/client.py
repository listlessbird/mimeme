from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from mimeme.inference.model import Annotation, Batch, BatchResult, Input

Progress = Callable[[str, float], Awaitable[None]]


@runtime_checkable
class Client(Protocol):
    async def annotate(self, input: Input, *, progress: Progress | None = None) -> Annotation: ...

    async def embed(self, batch: Batch, *, progress: Progress | None = None) -> BatchResult: ...

    async def ready(self) -> bool: ...

    async def close(self) -> None: ...
