from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import Engine, delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import shared.db as db_module
from shared.models import IndexBuild


def _async_url(sync_url: str) -> str:
    if sync_url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + sync_url[len("postgresql://") :]
    if sync_url.startswith("postgres://"):
        return "postgresql+asyncpg://" + sync_url[len("postgres://") :]
    return sync_url


@pytest.fixture()
async def async_session_factory(
    db_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    if db_engine.dialect.name != "postgresql":
        pytest.skip("async DB tests require PostgreSQL")

    sync_url = db_engine.url.render_as_string(hide_password=False)
    engine = create_async_engine(_async_url(sync_url), echo=False, future=True)
    factory = async_sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )

    monkeypatch.setattr(db_module, "get_async_session_factory", lambda: factory)

    yield factory

    await engine.dispose()


async def test_read_session_executes_database_work(
    async_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with db_module.read_session() as session:
        result = await session.execute(text("select 1"))

    assert result.scalar_one() == 1


def _delete_build(db_engine: Engine, version: str) -> None:
    with db_engine.begin() as connection:
        connection.execute(delete(IndexBuild).where(IndexBuild.version == version))


def _build_count(db_engine: Engine, version: str) -> int:
    with db_engine.connect() as connection:
        count = connection.scalar(
            select(func.count()).select_from(IndexBuild).where(IndexBuild.version == version)
        )
        assert count is not None
        return count


async def test_read_session_does_not_commit_database_work(
    db_engine: Engine,
    async_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    version = "test-async-read-session-no-commit"
    _delete_build(db_engine, version)

    async with db_module.read_session() as session:
        session.add(IndexBuild(version=version, is_active=False))
        await session.flush()

    assert _build_count(db_engine, version) == 0


async def test_write_session_commits_successful_database_work(
    db_engine: Engine,
    async_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    version = "test-async-write-session-commits"
    _delete_build(db_engine, version)

    async with db_module.write_session() as session:
        session.add(IndexBuild(version=version, is_active=False))

    assert _build_count(db_engine, version) == 1
    _delete_build(db_engine, version)


async def test_write_session_rolls_back_database_work_on_exception(
    db_engine: Engine,
    async_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    version = "test-async-write-session-rolls-back"
    _delete_build(db_engine, version)

    with pytest.raises(RuntimeError, match="boom"):
        async with db_module.write_session() as session:
            session.add(IndexBuild(version=version, is_active=False))
            await session.flush()
            raise RuntimeError("boom")

    assert _build_count(db_engine, version) == 0
