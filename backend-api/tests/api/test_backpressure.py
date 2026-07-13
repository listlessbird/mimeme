from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import Engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from structlog.testing import capture_logs

import shared.db as db_module
from api.middleware import register_middleware
from shared.config import settings


def _async_url(sync_url: str) -> str:
    if sync_url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + sync_url[len("postgresql://") :]
    if sync_url.startswith("postgres://"):
        return "postgresql+asyncpg://" + sync_url[len("postgres://") :]
    return sync_url


def _make_app() -> FastAPI:
    app = FastAPI()
    register_middleware(app)

    @app.get("/db")
    async def db_read() -> dict[str, bool]:
        async with db_module.read_session() as session:
            await session.execute(text("select 1"))
        return {"ok": True}

    @app.get("/db-sleep")
    async def db_sleep(seconds: float) -> dict[str, bool]:
        async with db_module.read_session() as session:
            await session.execute(text("select pg_sleep(:s)").bindparams(s=seconds))
        return {"ok": True}

    @app.get("/slow")
    async def slow() -> dict[str, bool]:
        await asyncio.sleep(0.5)
        return {"ok": True}

    return app


@pytest.fixture()
async def tiny_pool_factory(
    db_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    if db_engine.dialect.name != "postgresql":
        pytest.skip("backpressure tests require PostgreSQL")

    sync_url = db_engine.url.render_as_string(hide_password=False)
    engine = create_async_engine(
        _async_url(sync_url),
        echo=False,
        future=True,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.2,
    )
    db_module.instrument_pool(engine)
    factory = async_sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(db_module, "get_async_session_factory", lambda: factory)

    yield factory

    await engine.dispose()


@pytest.fixture()
async def backpressure_client(
    tiny_pool_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=_make_app(), raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_saturated_pool_returns_503_with_retry_after(
    backpressure_client: AsyncClient,
) -> None:
    with capture_logs() as logs:
        holder = asyncio.create_task(backpressure_client.get("/db-sleep", params={"seconds": 1.0}))
        await asyncio.sleep(0.3)

        starved = await backpressure_client.get("/db")
        held = await holder

    assert held.status_code == 200
    assert starved.status_code == 503
    assert starved.headers["Retry-After"] == "1"
    assert starved.json() == {"detail": "Server overloaded, retry shortly"}

    overloaded = [e for e in logs if e["event"] == "http_request" and e["status_code"] == 503]
    assert len(overloaded) == 1
    assert overloaded[0]["pool_wait_ms"] > 0


async def test_handler_exceeding_deadline_returns_504(
    backpressure_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "request_timeout_s", 0.1)

    response = await backpressure_client.get("/slow")

    assert response.status_code == 504
    assert response.json() == {"detail": "Request timed out"}


async def test_wide_event_emitted_once_per_request_with_db_timings(
    backpressure_client: AsyncClient,
) -> None:
    with capture_logs() as logs:
        response = await backpressure_client.get("/db")

    assert response.status_code == 200

    events = [e for e in logs if e["event"] == "http_request"]
    assert len(events) == 1

    event = events[0]
    assert event["method"] == "GET"
    assert event["route"] == "/db"
    assert event["status_code"] == 200
    assert event["duration_ms"] >= 0
    assert event["pool_wait_ms"] >= 0
    assert event["db_held_ms"] > 0
    assert event["pool_in_use"] >= 0
    assert event["client_key"]
    assert event["timed_out"] is False
