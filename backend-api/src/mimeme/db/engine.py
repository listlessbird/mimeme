from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from sqlalchemy import event
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import ConnectionPoolEntry

from mimeme.shared.config import DatabaseConfig


@dataclass
class DbRequestMetrics:
    pool_wait_ms: float = 0.0
    db_held_ms: float = 0.0


db_request_metrics: ContextVar[DbRequestMetrics | None] = ContextVar(
    "db_request_metrics", default=None
)


def begin_request_metrics() -> DbRequestMetrics:
    metrics = DbRequestMetrics()
    db_request_metrics.set(metrics)
    return metrics


def _on_checkout(
    dbapi_connection: DBAPIConnection,
    connection_record: ConnectionPoolEntry,
    connection_proxy: object,
) -> None:
    connection_record.info["checked_out_at"] = time.monotonic()


def _on_checkin(
    dbapi_connection: DBAPIConnection | None,
    connection_record: ConnectionPoolEntry,
) -> None:
    checked_out_at = connection_record.info.pop("checked_out_at", None)
    metrics = db_request_metrics.get()
    if metrics is not None and isinstance(checked_out_at, float):
        metrics.db_held_ms += (time.monotonic() - checked_out_at) * 1000


async def _acquire_connection(session: AsyncSession) -> None:
    metrics = db_request_metrics.get()
    if metrics is None:
        await session.connection()
        return

    started = time.monotonic()
    try:
        await session.connection()
    finally:
        metrics.pool_wait_ms += (time.monotonic() - started) * 1000


class Db:
    def __init__(self, config: DatabaseConfig) -> None:
        connect_args: dict[str, object] = {
            "statement_cache_size": config.statement_cache_size
        }
        if config.ssl_required:
            connect_args["ssl"] = True

        self._engine = create_async_engine(
            config.async_url_str,
            echo=False,
            future=True,
            pool_pre_ping=False,
            pool_recycle=240,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_timeout=config.pool_timeout_s,
            connect_args=connect_args,
        )
        event.listen(self._engine.sync_engine, "checkout", _on_checkout)
        event.listen(self._engine.sync_engine, "checkin", _on_checkin)

        self._sessions = async_sessionmaker(
            bind=self._engine, autoflush=False, expire_on_commit=False
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @asynccontextmanager
    async def read_session(self) -> AsyncGenerator[AsyncSession]:
        async with self._sessions() as session:
            await _acquire_connection(session)
            yield session

    @asynccontextmanager
    async def write_session(self) -> AsyncGenerator[AsyncSession]:
        async with self._sessions() as session:
            await _acquire_connection(session)
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def close(self) -> None:
        await self._engine.dispose()
