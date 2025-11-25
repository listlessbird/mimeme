from __future__ import annotations
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from .config import CFG


def get_engine():
    engine = create_engine(f"sqlite:///{CFG.db_path}", future=True)

    pragmas: tuple[tuple[str, str], ...] = (
        ("PRAGMA journal_mode", "WAL"),
        ("PRAGMA synchronous", "NORMAL"),
        ("PRAGMA temp_store", "MEMORY"),
        ("PRAGMA mmap_size", "30000000000"),
        ("PRAGMA page_size", "32768"),
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, connection_record):
        cur = dbapi_connection.cursor()
        try:
            for k, v in pragmas:
                cur.execute(f"{k}={v}")
        finally:
            cur.close()

    return engine


_engine = get_engine()
SessionLocal = sessionmaker(bind=_engine, future=True, expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
