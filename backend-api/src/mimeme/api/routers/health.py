from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, Response, status
from sqlalchemy import text

from mimeme.api.deps import get_temporal_client
from mimeme.api.models.errors import error_responses
from mimeme.api.models.health import HealthResponse
from mimeme.shared.db import get_async_engine
from mimeme.shared.services.api_storage import AsyncApiStorage
from mimeme.shared.services.storage import S3Config, get_artifact_s3_config, get_media_s3_config

router = APIRouter(tags=["Health"], responses=error_responses(429, 500))
log = structlog.get_logger()


async def _check_postgres() -> bool:
    try:
        engine = get_async_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        log.warning("healthcheck_postgres_failed", error=str(e))
        return False


async def _check_storage(role: str, config: S3Config) -> bool:
    try:
        return await AsyncApiStorage(config).bucket_exists()
    except Exception as e:
        log.warning("healthcheck_storage_failed", role=role, error=str(e))
        return False


async def _check_temporal() -> bool:
    try:
        client = await get_temporal_client()
        await client.service_client.check_health()
        return True
    except Exception as e:
        log.warning("healthcheck_temporal_failed", error=str(e))
        return False


@router.get(
    "/ready",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Dependency unavailable"}},
)
async def check_readiness(response: Response) -> HealthResponse:
    pg_ok, media_ok, artifact_ok = await asyncio.gather(
        _check_postgres(),
        _check_storage("media", get_media_s3_config()),
        _check_storage("artifact", get_artifact_s3_config()),
    )

    temporal_ok = await _check_temporal()

    healthy = pg_ok and media_ok and artifact_ok and temporal_ok

    if not healthy:
        log.warning(
            "healthcheck_degraded",
            postgres=pg_ok,
            media_storage=media_ok,
            artifact_storage=artifact_ok,
            temporal=temporal_ok,
        )

    response.status_code = status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="ok" if healthy else "degraded")
