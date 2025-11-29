from __future__ import annotations

from datetime import UTC, datetime

import structlog
from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from api.config import settings
from api.deps import DbSession, IndexManagerDep

router = APIRouter()
log = structlog.get_logger()


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime
    version: str
    environment: str

    database: str
    index: str
    index_version: str | None

    total_images: int
    indexed_images: int


@router.get("/health", response_model=HealthResponse)
async def health_check(
    db: DbSession,
    index_manager: IndexManagerDep,
) -> HealthResponse:
    db_status = "healthy"
    total_images = 0
    try:
        result = db.execute(text("SELECT COUNT(*) FROM images"))
        total_images = result.scalar() or 0
    except Exception as e:
        log.error("database_health_check_failed", error=str(e))
        db_status = "unhealthy"

    index_status = "healthy" if index_manager.is_loaded else "not_loaded"
    indexed_images = index_manager.num_vectors if index_manager.is_loaded else 0

    overall_status = "healthy"
    if db_status != "healthy":
        overall_status = "degraded"
    if index_status == "not_loaded" and total_images > 0:
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(UTC),
        version="0.2.0",
        environment=settings.app_env,
        database=db_status,
        index=index_status,
        index_version=index_manager.active_version,
        total_images=total_images,
        indexed_images=indexed_images,
    )
