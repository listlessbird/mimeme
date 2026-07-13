from collections.abc import AsyncGenerator, Generator, Iterator
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from shared.config import settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(
        settings.db_url_str,
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
        yield session


@asynccontextmanager
async def write_session() -> AsyncGenerator[AsyncSession]:
    factory = get_async_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@lru_cache(maxsize=1)
def get_async_engine() -> AsyncEngine:
    connect_args: dict[str, object] = {"statement_cache_size": settings.db_statement_cache_size}

    if settings.db_ssl_required:
        connect_args["ssl"] = True

    return create_async_engine(
        settings.async_db_url_str,
        echo=False,
        future=True,
        pool_pre_ping=False,
        pool_recycle=240,
        pool_size=settings.db_pool_size_async,
        max_overflow=settings.db_max_overflow_async,
        pool_timeout=settings.db_pool_timeout_s,
        connect_args=connect_args,
    )
