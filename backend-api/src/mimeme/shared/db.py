import time
from collections.abc import AsyncGenerator, Generator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import ConnectionPoolEntry, PoolProxiedConnection

from mimeme.shared.config import settings


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
    connection_proxy: PoolProxiedConnection,
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


def instrument_pool(engine: AsyncEngine) -> None:
    event.listen(engine.sync_engine, "checkout", _on_checkout)
    event.listen(engine.sync_engine, "checkin", _on_checkin)


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


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(
        settings.database.url_str,
        echo=False,
        future=True,
        pool_pre_ping=False,
        pool_recycle=240,
        pool_size=5,
        max_overflow=10,
        connect_args={
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 3,
        },
    )


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(
        bind=get_engine(), autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )


@contextmanager
def session_scope() -> Generator[Session]:
    factory = get_session_factory()
    session = factory()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def read_session_scope() -> Generator[Session]:
    factory = get_session_factory()
    session = factory()

    try:
        yield session
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    factory = get_session_factory()
    session = factory()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@lru_cache(maxsize=1)
def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=get_async_engine(), autoflush=False, expire_on_commit=False)


@asynccontextmanager
async def read_session() -> AsyncGenerator[AsyncSession]:
    factory = get_async_session_factory()
    async with factory() as session:
        await _acquire_connection(session)
        yield session


@asynccontextmanager
async def write_session() -> AsyncGenerator[AsyncSession]:
    factory = get_async_session_factory()
    async with factory() as session:
        await _acquire_connection(session)
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@lru_cache(maxsize=1)
def get_async_engine() -> AsyncEngine:
    connect_args: dict[str, object] = {
        "statement_cache_size": settings.database.statement_cache_size
    }

    if settings.database.ssl_required:
        connect_args["ssl"] = True

    engine = create_async_engine(
        settings.database.async_url_str,
        echo=False,
        future=True,
        pool_pre_ping=False,
        pool_recycle=240,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout_s,
        connect_args=connect_args,
    )
    instrument_pool(engine)
    return engine
