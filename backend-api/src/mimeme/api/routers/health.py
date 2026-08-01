from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, Response, status
from sqlalchemy import text
from temporalio.client import Client

from mimeme import search, storage
from mimeme.api.deps import (
    ArtifactStorageDep,
    DbDep,
    MediaStorageDep,
    SearchDep,
    TemporalClientDep,
)
from mimeme.api.models.errors import error_responses
from mimeme.api.models.health import HealthResponse
from mimeme.db import Db

router = APIRouter(tags=["Health"], responses=error_responses(429, 500))
log = structlog.get_logger()


async def _check_postgres(db: Db) -> bool:
    try:
        async with db.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        log.warning("healthcheck_postgres_failed", error=str(e))
        return False


async def _check_storage(role: str, store: storage.Store) -> bool:
    try:
        await store.probe()
        return True
    except Exception as e:
        log.warning("healthcheck_storage_failed", role=role, error=str(e))
        return False


async def _check_temporal(client: Client) -> bool:
    try:
        await client.service_client.check_health()
        return True
    except Exception as e:
        log.warning("healthcheck_temporal_failed", error=str(e))
        return False


async def _check_search(client: search.Client) -> bool:
    try:
        return (await client.status()).ready
    except search.Error as e:
        log.warning("healthcheck_search_failed", error=str(e))
        return False


@router.get(
    "/ready",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Dependency unavailable"}},
)
async def check_readiness(
    response: Response,
    db: DbDep,
    temporal: TemporalClientDep,
    media: MediaStorageDep,
    artifacts: ArtifactStorageDep,
    search_client: SearchDep,
) -> HealthResponse:
    pg_ok, media_ok, artifact_ok, temporal_ok, search_ok = await asyncio.gather(
        _check_postgres(db),
        _check_storage("media", media),
        _check_storage("artifact", artifacts),
        _check_temporal(temporal),
        _check_search(search_client),
    )

    healthy = pg_ok and media_ok and artifact_ok and temporal_ok and search_ok

    if not healthy:
        log.warning(
            "healthcheck_degraded",
            postgres=pg_ok,
            media_storage=media_ok,
            artifact_storage=artifact_ok,
            temporal=temporal_ok,
            search=search_ok,
        )

    response.status_code = status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="ok" if healthy else "degraded")
