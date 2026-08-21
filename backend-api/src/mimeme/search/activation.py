from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from mimeme.search.client import Activation
from mimeme.search.error import Unavailable
from mimeme.search.model import Load, Loaded, Status

Commit = Callable[[Loaded], Awaitable[None]]

_RETRY_ATTEMPTS = 5
_RETRY_BASE_DELAY_S = 1.0


async def _with_retry[T](operation: Callable[[], Awaitable[T]]) -> T:
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return await operation()
        except Unavailable:
            if attempt == _RETRY_ATTEMPTS - 1:
                raise
            await asyncio.sleep(_RETRY_BASE_DELAY_S * 2**attempt)


async def activate(generation: Load, *, activation: Activation, commit: Commit) -> Status:
    loaded = await activation.load(generation)
    status = await activation.switch(loaded.version)
    try:
        await commit(loaded)
    except BaseException:
        await activation.rollback(loaded.version)
        raise
    return status


async def reconcile(generation: Load, *, activation: Activation) -> Status:
    status = await _with_retry(activation.status)
    if status.serving_version == generation.version:
        return status
    loaded = await _with_retry(lambda: activation.load(generation))
    return await _with_retry(lambda: activation.switch(loaded.version))
