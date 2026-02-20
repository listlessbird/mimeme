from __future__ import annotations

from collections.abc import Generator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from api import deps as api_deps
from api import main as api_main
from shared.config import settings
from shared.models import Base


@dataclass
class FakeStorage:
    presigned_calls: list[tuple[str, int]] = field(default_factory=list)
    delete_calls: list[str] = field(default_factory=list)

    def generate_presigned_url(self, key: str, expiration: int) -> str:
        self.presigned_calls.append((key, expiration))
        return f"https://example.invalid/{key}"

    def delete(self, key: str) -> None:
        self.delete_calls.append(key)


class FakeWorkflowHandle:
    def __init__(self, workflow_id: str, cancelled: list[str]) -> None:
        self.workflow_id = workflow_id
        self._cancelled = cancelled

    async def cancel(self) -> None:
        self._cancelled.append(self.workflow_id)


@dataclass
class FakeTemporalClient:
    started_workflows: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)
    cancelled_workflows: list[str] = field(default_factory=list)

    async def start_workflow(self, *args: Any, **kwargs: Any) -> None:
        self.started_workflows.append((args, kwargs))

    def get_workflow_handle(self, workflow_id: str) -> FakeWorkflowHandle:
        return FakeWorkflowHandle(workflow_id=workflow_id, cancelled=self.cancelled_workflows)


@dataclass
class FakeIndexManager:
    is_loaded: bool = True
    active_version: str | None = "v-test-index"
    num_vectors: int = 2


@pytest.fixture
def fake_storage() -> FakeStorage:
    return FakeStorage()


@pytest.fixture
def fake_temporal_client() -> FakeTemporalClient:
    return FakeTemporalClient()


@pytest.fixture
def fake_index_manager() -> FakeIndexManager:
    return FakeIndexManager()


@pytest.fixture
def session_factory() -> sessionmaker[Session]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )


@pytest.fixture
def app(
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker[Session],
    fake_storage: FakeStorage,
    fake_temporal_client: FakeTemporalClient,
    fake_index_manager: FakeIndexManager,
):
    @asynccontextmanager
    async def _no_lifespan(_app):
        yield

    monkeypatch.setattr(api_main, "lifespan", _no_lifespan)
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "api_key_admin", "admin-test-key")
    monkeypatch.setattr(settings, "api_key_readonly", "readonly-test-key")

    fastapi_app = api_main.create_app()

    def _get_db() -> Generator[Session, None, None]:
        db = session_factory()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    async def _get_temporal():
        return fake_temporal_client

    def _get_storage():
        return fake_storage

    def _get_index_manager():
        return fake_index_manager

    fastapi_app.dependency_overrides[api_deps.get_db] = _get_db
    fastapi_app.dependency_overrides[api_deps.get_temporal_client] = _get_temporal
    fastapi_app.dependency_overrides[api_deps.get_storage] = _get_storage
    fastapi_app.dependency_overrides[api_deps.get_index_manager] = _get_index_manager
    fastapi_app.state.session_factory = session_factory
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


@pytest.fixture
def admin_headers() -> dict[str, str]:
    return {"X-API-Key": "admin-test-key"}


@pytest.fixture
def readonly_headers() -> dict[str, str]:
    return {"X-API-Key": "readonly-test-key"}
