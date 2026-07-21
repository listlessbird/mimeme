from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from mimeme.db.schema import Image
from tests.factories import create_image

T = TypeVar("T")
SyncSeedRunner = Callable[[Callable[[Session], T]], Awaitable[T]]


async def test_sync_factory_seed_rows_are_visible_to_async_session(
    run_sync_seed: SyncSeedRunner,
    async_db_session: AsyncSession,
) -> None:
    image = await run_sync_seed(
        lambda session: create_image(
            session=session,
            original_filename="async-fixture-visible.jpg",
        )
    )

    result = await async_db_session.execute(select(Image).where(Image.id == image.id))

    assert result.scalar_one().original_filename == "async-fixture-visible.jpg"


async def test_async_db_session_rolls_back_factory_seed_from_previous_test(
    async_db_session: AsyncSession,
) -> None:
    count = await async_db_session.scalar(
        select(func.count())
        .select_from(Image)
        .where(Image.original_filename == "async-fixture-visible.jpg")
    )

    assert count == 0
