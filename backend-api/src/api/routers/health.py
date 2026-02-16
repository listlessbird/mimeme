from __future__ import annotations

import structlog
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from shared.config import settings
from shared.db import get_engine
from shared.services.storage import get_storage_service

router = APIRouter()
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
        storage = get_storage_service()
        storage.client.head_bucket(Bucket=storage.bucket)
        return True
    except Exception as e:
        log.warning("healthcheck_s3_failed", error=str(e))
        return False


async def _check_temporal() -> bool:
    try:
        client = await Client.connect(
            settings.temporal_host,
            data_converter=pydantic_data_converter,
        )
        await client.service_client.check_health()
        return True
    except Exception as e:
        log.warning("healthcheck_temporal_failed", error=str(e))
        return False


@router.get("/live")
async def healthcheck() -> JSONResponse:
    pg_ok = _check_postgres()
    s3_ok = _check_s3()
    temporal_ok = await _check_temporal()

    healthy = pg_ok and s3_ok and temporal_ok

    if not healthy:
        log.warning(
            "healthcheck_degraded",
            postgres=pg_ok,
            s3=s3_ok,
            temporal=temporal_ok,
        )

    status_code = status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(status_code=status_code, content={"status": "ok" if healthy else "degraded"})
