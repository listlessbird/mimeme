import time
from typing import Annotated

import structlog
from fastapi import APIRouter, HTTPException, Query
from prometheus_client import Counter, Histogram

from api.deps import DbSession, SearchServiceDep
from api.models.search import SearchResponse

log = structlog.get_logger()
router = APIRouter()

SEARCH_REQUESTS = Counter("search_requests_total", "Total search requests", ["status"])
SEARCH_LATENCY = Histogram(
    "search_latency_seconds",
    "Search latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)


@router.get("", response_model=SearchResponse)
async def search(
    q: Annotated[str, Query(min_length=1, max_length=200, description="Search query")],
    search_service: SearchServiceDep,
    db: DbSession,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> SearchResponse:
    start_time = time.perf_counter()

    try:
        results = search_service.search(query=q, limit=limit + offset, db=db)

        paginated = results[offset : offset + limit]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        SEARCH_REQUESTS.labels(status="success").inc()
        SEARCH_LATENCY.observe(elapsed_ms / 1000)
        log.info(
            "search_completed",
            query=q,
            num_results=len(paginated),
            total=len(results),
            latency_ms=elapsed_ms,
        )

        return SearchResponse(
            query=q,
            results=paginated,
            total=len(results),
            limit=limit,
            offset=offset,
            search_time_ms=round(elapsed_ms, 2),
            index_version=search_service.index_manager.active_version,
        )

    except ValueError as e:
        SEARCH_REQUESTS.labels(status="error").inc()
        log.warning("search_failed", query=q, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        SEARCH_REQUESTS.labels(status="error").inc()
        log.exception("search_error", query=q)
        raise HTTPException(status_code=500, detail="Search failed")


@router.get("/similar/{image_id}", response_model=SearchResponse)
async def find_similar(
    image_id: int,
    search_service: SearchServiceDep,
    db: DbSession,
    limit: int = Query(default=20, ge=1, le=100),
) -> SearchResponse:
    start_time = time.perf_counter()

    try:
        results = search_service.find_similar(
            image_id=image_id,
            limit=limit + 1,  # +1 to exclude self
            db=db,
        )

        results = [r for r in results if r.id != image_id][:limit]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SearchResponse(
            query=f"similar_to:{image_id}",
            results=results,
            total=len(results),
            limit=limit,
            offset=0,
            search_time_ms=round(elapsed_ms, 2),
            index_version=search_service.index_manager.active_version,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
