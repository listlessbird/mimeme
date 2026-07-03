from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, delete, func, select, text
from sqlalchemy.orm import Session, sessionmaker

import shared.db as db_module
from shared.models import IndexBuild


@pytest.fixture()
def db_session_factory(
    db_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> Iterator[sessionmaker[Session]]:
    factory = sessionmaker(
        bind=db_engine,
        autoflush=False,
        autocommit=False,
        future=True,
        expire_on_commit=False,
    )
    monkeypatch.setattr(db_module, "get_session_factory", lambda: factory)
    yield factory


def _delete_build(db_engine: Engine, version: str) -> None:
    with db_engine.begin() as connection:
        connection.execute(delete(IndexBuild).where(IndexBuild.version == version))


def _build_count(factory: sessionmaker[Session], version: str) -> int:
    with factory() as session:
        count = session.scalar(
            select(func.count()).select_from(IndexBuild).where(IndexBuild.version == version)
        )
        assert count is not None
        return count


def test_read_session_scope_yields_working_session(
    db_session_factory: sessionmaker[Session],
) -> None:
    with db_module.read_session_scope() as session:
        assert session.execute(text("select 1")).scalar_one() == 1


def test_read_session_scope_does_not_commit(
    db_engine: Engine,
    db_session_factory: sessionmaker[Session],
) -> None:
    version = "test-read-session-no-commit"
    _delete_build(db_engine, version)

    with db_module.read_session_scope() as session:
        session.add(IndexBuild(version=version, is_active=False))
        session.flush()

    assert _build_count(db_session_factory, version) == 0


def test_session_scope_still_commits(
    db_engine: Engine,
    db_session_factory: sessionmaker[Session],
) -> None:
    version = "test-session-scope-commits"
    _delete_build(db_engine, version)

    with db_module.session_scope() as session:
        session.add(IndexBuild(version=version, is_active=False))

    assert _build_count(db_session_factory, version) == 1
    _delete_build(db_engine, version)
