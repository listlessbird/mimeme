import asyncio
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request

from api.auth import ReadonlyRequired
from api.deps import IndexManagerDep
from api.models.search import SearchResponse
from api.rate_limit import SEARCH_LIMIT, limiter
from domain.search_index import (
    SearchExecutionFailedError,
    SearchImageNotFoundError,
    SearchIndexExecution,
    SearchIndexLoadError,
    SearchIndexUnavailableError,
    SearchInvalidRequestError,
    SearchQueryEncodingError,
)

router = APIRouter(prefix="/search", tags=["Search"])


@router.get("", response_model=SearchResponse)
@limiter.limit(SEARCH_LIMIT)
async def search(
    request: Request,
    _auth: ReadonlyRequired,
    index_manager: IndexManagerDep,
    q: Annotated[str, Query(min_length=1, max_length=200, description="Search query")],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    mode: Annotated[
        Literal["hybrid"] | None,
        Query(description="Optional mode. Omit for image search; use 'hybrid' to fuse image and text indexes."),
    ] = None,
) -> SearchResponse:
    resolved_mode: Literal["image", "hybrid"] = "hybrid" if mode == "hybrid" else "image"
    try:
        page = await asyncio.to_thread(
            SearchIndexExecution(index_manager).search,
            query=q,
            limit=limit,
            offset=offset,
            mode=resolved_mode,
        )
        return SearchResponse.model_construct(
            query=page.query,
            results=page.results,
            total=page.total,
            limit=page.limit,
            offset=page.offset,
            search_time_ms=page.search_time_ms,
            index_version=page.index_version,
        )
    except SearchIndexUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except SearchIndexLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except SearchQueryEncodingError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except SearchInvalidRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except SearchExecutionFailedError:
        raise HTTPException(status_code=500, detail="Search failed")


@router.get("/similar/{image_id}", response_model=SearchResponse)
@limiter.limit(SEARCH_LIMIT)
async def find_similar(
    request: Request,
    _auth: ReadonlyRequired,
    image_id: int,
    index_manager: IndexManagerDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> SearchResponse:
    try:
        page = await asyncio.to_thread(
            SearchIndexExecution(index_manager).find_similar,
            image_id=image_id,
            limit=limit,
        )
        return SearchResponse.model_construct(
            query=page.query,
            results=page.results,
            total=page.total,
            limit=page.limit,
            offset=page.offset,
            search_time_ms=page.search_time_ms,
            index_version=page.index_version,
        )
    except SearchIndexUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except SearchIndexLoadError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except SearchImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except SearchExecutionFailedError:
        raise HTTPException(status_code=500, detail="Search failed")
