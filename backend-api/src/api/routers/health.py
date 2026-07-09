from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, Response, status
from sqlalchemy import text

from api.deps import get_storage_probe, get_temporal_client
from api.models.errors import error_responses
from api.models.health import HealthResponse
from shared.db import get_engine

router = APIRouter(tags=["Health"], responses=error_responses(429, 500))
log = structlog.get_logger()


def _check_postgres() -> bool:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        log.warning("healthcheck_postgres_failed", error=str(e))
        return False


def _check_s3() -> bool:
    try:
        return get_storage_probe().bucket_exists()
    except Exception as e:
        log.warning("healthcheck_s3_failed", error=str(e))
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
    pg_ok, s3_ok = await asyncio.gather(
        asyncio.to_thread(_check_postgres), asyncio.to_thread(_check_s3)
    )

    temporal_ok = await _check_temporal()

    healthy = pg_ok and s3_ok and temporal_ok

    if not healthy:
        log.warning(
            "healthcheck_degraded",
            postgres=pg_ok,
            s3=s3_ok,
            temporal=temporal_ok,
        )

    response.status_code = status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="ok" if healthy else "degraded")
