from collections.abc import Generator, Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from mimeme.shared.runtime import settings

# FROZEN synchronous database access for unconverted activity/worker feature
# code. No new callers, no new methods. Owners migrate to db.Db (async):
# plans 003, 005, 006, 008. Plan 009 deletes this module and psycopg2.


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
