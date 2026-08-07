from __future__ import annotations

import os
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import TypeVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Connection, Engine, create_engine, event, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import Session

from mimeme.db.schema import Base

T = TypeVar("T")


DEFAULT_TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/mimeme_test"


def _sync_test_url() -> str:
    url = os.environ.get("TEST_DB_URL", DEFAULT_TEST_DB_URL)
    for prefix in ("postgresql+asyncpg://", "postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def _build_test_engine() -> Engine:
    engine = create_engine(_sync_test_url(), echo=False, future=True, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine


def _async_test_url(engine: Engine) -> str:
    sync_url = engine.url.render_as_string(hide_password=False)
    for prefix in ("postgresql+psycopg://", "postgresql://", "postgres://"):
        if sync_url.startswith(prefix):
            return "postgresql+asyncpg://" + sync_url[len(prefix) :]
    return sync_url


@pytest.fixture(scope="session")
def db_engine() -> Iterator[Engine]:
    engine = _build_test_engine()
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture()
def db_session(db_engine: Engine) -> Iterator[Session]:
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, expire_on_commit=False)
    nested = connection.begin_nested()

    @event.listens_for(session, "after_transaction_end")
    def _restart_savepoint(sess: Session, txn: object) -> None:
        nonlocal nested
        if not connection.closed and not connection.invalidated:
            if not connection.in_nested_transaction():
                nested = connection.begin_nested()

    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture()
async def async_db_engine(db_engine: Engine) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(_async_test_url(db_engine), echo=False, future=True)
    yield engine
    await engine.dispose()


@pytest.fixture()
async def async_db_connection(async_db_engine: AsyncEngine) -> AsyncIterator[AsyncConnection]:
    connection = await async_db_engine.connect()
    transaction = await connection.begin()
    yield connection
    await transaction.rollback()
    await connection.close()


@pytest.fixture()
async def async_db_session(async_db_connection: AsyncConnection) -> AsyncIterator[AsyncSession]:
    session = AsyncSession(
        bind=async_db_connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    yield session
    await session.close()


@pytest.fixture()
def run_sync_seed(
    async_db_connection: AsyncConnection,
) -> Callable[[Callable[[Session], T]], Awaitable[T]]:
    async def _run(seed: Callable[[Session], T]) -> T:
        def _inside(sync_connection: Connection) -> T:
            session = Session(
                bind=sync_connection,
                expire_on_commit=False,
                join_transaction_mode="rollback_only",
            )
            try:
                result = seed(session)
                session.flush()
                return result
            finally:
                session.close()

        return await async_db_connection.run_sync(_inside)

    return _run


@pytest.fixture()
def mock_temporal() -> AsyncMock:
    client = AsyncMock()
    client.start_workflow = AsyncMock(return_value=MagicMock(id="mock-workflow-id"))
    handle = MagicMock()
    handle.cancel = AsyncMock()
    client.get_workflow_handle = MagicMock(return_value=handle)
    return client
