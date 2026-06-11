"""Root test fixtures for the mimeme backend API test suite.

Provides:
- Test database with transaction-rollback isolation per test
- FastAPI test client with all dependencies overridden
- Mock fixtures for external services (S3, Temporal, FAISS)

When PostgreSQL is available (TEST_DB_URL env var), it uses a real PG test database.
Otherwise, falls back to an in-process SQLite database for CI environments
without Docker.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

# Force development mode + disable text encoder preloading before importing app code
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("PRELOAD_TEXT_ENCODER_ON_STARTUP", "false")

from shared.models.orm import Base  # noqa: E402


def _build_test_engine() -> Engine:
    """Create a test engine.  Tries PostgreSQL first, falls back to SQLite."""
    pg_url = os.environ.get("TEST_DB_URL")
    if pg_url:
        return create_engine(pg_url, echo=False, future=True, pool_pre_ping=True)

    # Try default PG URL
    default_pg = "postgresql://postgres:postgres@localhost:5432/mimeme_test"
    try:
        engine = create_engine(default_pg, echo=False, future=True, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception:
        pass

    # Fall back to SQLite (in-memory, shared across connections)
    return create_engine(
        "sqlite://",
        echo=False,
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def db_engine() -> Iterator[Engine]:
    """Create tables once for the entire test session."""
    engine = _build_test_engine()

    # SQLite doesn't support PostgreSQL enums natively, but SQLAlchemy's
    # Enum type auto-adapts to VARCHAR on SQLite.
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture()
def db_session(db_engine: Engine) -> Iterator[Session]:
    """Per-test session wrapped in a transaction that rolls back on teardown.

    Activities and routes that call ``session.commit()`` hit the savepoint,
    **not** the real transaction.  The outer rollback undoes everything so
    tests stay fully isolated.
    """
    is_sqlite = "sqlite" in str(db_engine.url)

    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, expire_on_commit=False)

    if not is_sqlite:
        # Savepoint-based isolation for PostgreSQL
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
def _patch_session_scope(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch ``shared.db.session_scope`` so that activities under test
    use the rollback-protected test session instead of a real connection.
    """

    @contextmanager
    def _test_session_scope() -> Iterator[Session]:
        yield db_session
        db_session.flush()

    monkeypatch.setattr("shared.db.session_scope", _test_session_scope)

    # Also patch at every import site so local references pick up the override
    for module_path in [
        "activities.workflow_state.activities.session_scope",
        "activities.storage.activities.session_scope",
        "activities.indexing.activities.session_scope",
        "activities.ingestion.activities.session_scope",
    ]:
        monkeypatch.setattr(module_path, _test_session_scope)

    def _test_get_db() -> Iterator[Session]:
        yield db_session

    monkeypatch.setattr("shared.db.get_db", _test_get_db)


# ---------------------------------------------------------------------------
# Mock fixtures for external services
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_storage() -> MagicMock:
    """Mock StorageService — no real S3 calls."""
    storage = MagicMock()
    storage.generate_presigned_url.return_value = "https://mock-s3/presigned"
    storage.upload_file.return_value = "mock-etag"
    storage.upload_bytes.return_value = "mock-etag"
    storage.delete.return_value = None
    storage.ensure_bucket_exists.return_value = None
    storage.build_image_key.return_value = "images/test/abc123.jpg"
    storage.build_embedding_key.return_value = "embeddings/test/abc123.npy"
    storage.client = MagicMock()
    storage.client.head_bucket.return_value = {}
    storage.bucket = "test-bucket"
    return storage


@pytest.fixture()
def mock_temporal() -> AsyncMock:
    """Mock Temporal client — captures workflow starts."""
    client = AsyncMock()
    client.start_workflow = AsyncMock(return_value=MagicMock(id="mock-workflow-id"))

    handle = MagicMock()
    handle.cancel = AsyncMock()
    client.get_workflow_handle = MagicMock(return_value=handle)
    return client


@pytest.fixture()
def mock_index_manager() -> MagicMock:
    """Mock FaissIndexManager — no real FAISS index needed."""
    manager = MagicMock()
    manager.is_loaded = True
    manager.is_text_loaded = False
    manager.active_version = "v1-test"
    manager.num_vectors = 100
    manager.has_text_index.return_value = False
    manager.search.return_value = []
    manager.search_text.return_value = []
    manager.get_vector_by_image_id.return_value = None
    manager.load_active_index.return_value = None
    return manager


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(
    db_session: Session,
    mock_storage: MagicMock,
    mock_temporal: AsyncMock,
    mock_index_manager: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    """httpx-backed test client with all heavy dependencies swapped out.

    * Lifespan is **disabled** (it connects to real Postgres/S3/Temporal).
    * DB session uses the savepoint-wrapped test session.
    * Storage, Temporal, and FAISS are all mocks.
    """
    from contextlib import asynccontextmanager

    from api.deps import get_db, get_index_manager, get_storage, get_temporal_client
    from api.main import create_app

    @asynccontextmanager
    async def _noop_lifespan(app: object):
        yield

    app = create_app()
    # Replace the real lifespan (which connects to Postgres/S3/Temporal)
    # with a no-op so the test client doesn't hit external services.
    app.router.lifespan_context = _noop_lifespan

    def _override_db() -> Iterator[Session]:
        yield db_session

    async def _override_temporal() -> AsyncMock:
        return mock_temporal

    def _override_storage() -> MagicMock:
        return mock_storage

    def _override_index_manager() -> MagicMock:
        return mock_index_manager

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_temporal_client] = _override_temporal
    app.dependency_overrides[get_storage] = _override_storage
    app.dependency_overrides[get_index_manager] = _override_index_manager

    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc
