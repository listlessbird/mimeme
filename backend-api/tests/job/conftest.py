from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import Engine
from sqlalchemy.ext.asyncio import AsyncConnection, async_sessionmaker, create_async_engine

from mimeme.db import Db


class SavepointDb(Db):
    """A ``Db`` bound to one savepoint-isolated connection so every operation
    still opens its own ``AsyncSession`` while the outer transaction rolls the
    whole test back."""

    def __init__(self, connection: AsyncConnection) -> None:
        self._engine = None  # type: ignore[assignment]
        self._sessions = async_sessionmaker(
            bind=connection,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )

    async def close(self) -> None:
        return None


class PoolDb(Db):
    """A ``Db`` over a real connection pool so concurrent operations acquire
    distinct connections and observe real row-level locking."""

    def __init__(self, async_url: str) -> None:
        self._engine = create_async_engine(async_url, echo=False, future=True, pool_size=5)
        self._sessions = async_sessionmaker(bind=self._engine, expire_on_commit=False)


def _async_url(engine: Engine) -> str:
    sync_url = engine.url.render_as_string(hide_password=False)
    for prefix in ("postgresql://", "postgres://"):
        if sync_url.startswith(prefix):
            return "postgresql+asyncpg://" + sync_url[len(prefix) :]
    return sync_url


@pytest.fixture()
def job_db(async_db_connection: AsyncConnection) -> SavepointDb:
    return SavepointDb(async_db_connection)


@pytest.fixture()
async def pool_db(db_engine: Engine) -> AsyncIterator[PoolDb]:
    if db_engine.dialect.name != "postgresql":
        pytest.skip("row-locking tests require PostgreSQL")
    db = PoolDb(_async_url(db_engine))
    yield db
    await db.close()
