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
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import BinaryIO, TypeVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy import Connection, Engine, create_engine, event, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

# Force development mode + disable text encoder preloading before importing app code
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("PRELOAD_TEXT_ENCODER_ON_STARTUP", "false")

from shared.models.orm import Base  # noqa: E402

T = TypeVar("T")


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


def _async_test_url(engine: Engine) -> str:
    sync_url = engine.url.render_as_string(hide_password=False)
    if sync_url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + sync_url[len("postgresql://") :]
    if sync_url.startswith("postgres://"):
        return "postgresql+asyncpg://" + sync_url[len("postgres://") :]
    return sync_url


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
async def async_db_engine(db_engine: Engine) -> AsyncIterator[AsyncEngine]:
    if db_engine.dialect.name != "postgresql":
        pytest.skip("async DB fixtures require PostgreSQL")

    engine = create_async_engine(_async_test_url(db_engine), echo=False, future=True)

    yield engine

    await engine.dispose()


@pytest.fixture()
async def async_db_connection(
    async_db_engine: AsyncEngine,
) -> AsyncIterator[AsyncConnection]:
    connection = await async_db_engine.connect()
    transaction = await connection.begin()

    yield connection

    await transaction.rollback()
    await connection.close()


@pytest.fixture()
async def async_db_session(
    async_db_connection: AsyncConnection,
) -> AsyncIterator[AsyncSession]:
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


@pytest.fixture()
def _patch_domain_session_scope(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def _test_session_scope() -> Iterator[Session]:
        yield db_session
        db_session.flush()

    monkeypatch.setattr("shared.db.session_scope", _test_session_scope)
    monkeypatch.setattr("shared.db.read_session_scope", _test_session_scope)


@pytest.fixture()
def _patch_async_domain_session_scope(
    async_db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @asynccontextmanager
    async def _test_read_session() -> AsyncIterator[AsyncSession]:
        yield async_db_session

    @asynccontextmanager
    async def _test_write_session() -> AsyncIterator[AsyncSession]:
        yield async_db_session
        await async_db_session.flush()

    monkeypatch.setattr("shared.db.read_session", _test_read_session)
    monkeypatch.setattr("shared.db.write_session", _test_write_session)


# ---------------------------------------------------------------------------
# Mock fixtures for external services
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UploadBytesCall:
    data: bytes | BinaryIO
    key: str
    content_type: str


class FakeApiStorage:
    def __init__(self) -> None:
        self.presigned: list[tuple[str, int]] = []
        self.uploaded: list[UploadBytesCall] = []
        self.deleted: list[str] = []
        self.existing_keys: set[str] = set()

    def presign(self, key: str, expiration: int = 3600) -> str:
        self.presigned.append((key, expiration))
        return "https://mock-s3/presigned"

    async def upload_bytes(
        self, data: bytes | BinaryIO, key: str, content_type: str = "application/octet-stream"
    ) -> str:
        self.uploaded.append(UploadBytesCall(data=data, key=key, content_type=content_type))
        self.existing_keys.add(key)
        return "mock-etag"

    async def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.existing_keys.discard(key)

    async def exists(self, key: str) -> bool:
        return key in self.existing_keys


@pytest.fixture()
def api_storage() -> FakeApiStorage:
    return FakeApiStorage()


@pytest.fixture()
def mock_storage() -> MagicMock:
    """Mock StorageService — no real S3 calls."""
    storage = MagicMock()
    storage.presign.return_value = "https://mock-s3/presigned"
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
    _patch_domain_session_scope: None,
    api_storage: FakeApiStorage,
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

    from api.deps import get_index_manager, get_storage, get_temporal_client
    from api.main import create_app

    @asynccontextmanager
    async def _noop_lifespan(app: object):
        yield

    app = create_app()
    # Replace the real lifespan (which connects to Postgres/S3/Temporal)
    # with a no-op so the test client doesn't hit external services.
    app.router.lifespan_context = _noop_lifespan

    async def _override_temporal() -> AsyncMock:
        return mock_temporal

    def _override_storage() -> FakeApiStorage:
        return api_storage

    def _override_index_manager() -> MagicMock:
        return mock_index_manager

    app.dependency_overrides[get_temporal_client] = _override_temporal
    app.dependency_overrides[get_storage] = _override_storage
    app.dependency_overrides[get_index_manager] = _override_index_manager

    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc


@pytest.fixture()
async def async_client(
    db_session: Session,
    _patch_domain_session_scope: None,
    api_storage: FakeApiStorage,
    mock_temporal: AsyncMock,
    mock_index_manager: MagicMock,
) -> AsyncIterator[AsyncClient]:
    from contextlib import asynccontextmanager

    from api.deps import get_index_manager, get_storage, get_temporal_client
    from api.main import create_app

    @asynccontextmanager
    async def _noop_lifespan(app: object):
        yield

    app = create_app()
    app.router.lifespan_context = _noop_lifespan

    async def _override_temporal() -> AsyncMock:
        return mock_temporal

    def _override_storage() -> FakeApiStorage:
        return api_storage

    def _override_index_manager() -> MagicMock:
        return mock_index_manager

    app.dependency_overrides[get_temporal_client] = _override_temporal
    app.dependency_overrides[get_storage] = _override_storage
    app.dependency_overrides[get_index_manager] = _override_index_manager

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
