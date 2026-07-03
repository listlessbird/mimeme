from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import cast

from sqlalchemy import Engine, create_engine
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
def session_scope() -> Iterator[Session]:
    factory = cast(sessionmaker[Session], get_session_factory())
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
def read_session_scope() -> Iterator[Session]:
    factory = cast(sessionmaker[Session], get_session_factory())
    session = factory()

    try:
        yield session
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    factory = cast(sessionmaker[Session], get_session_factory())
    session = factory()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
