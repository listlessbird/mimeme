from __future__ import annotations

from collections.abc import Awaitable, Callable

from mimeme.search.client import Activation
from mimeme.search.model import Load, Loaded, Status

Commit = Callable[[Loaded], Awaitable[None]]


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
    status = await activation.status()
    if status.serving_version == generation.version:
        return status
    loaded = await activation.load(generation)
    return await activation.switch(loaded.version)
