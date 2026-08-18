from __future__ import annotations

import time
from typing import Annotated, Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi import Query as HttpQuery

from mimeme import search
from mimeme.api.auth import ReadonlyRequired
from mimeme.api.models.errors import error_responses
from mimeme.api.rate_limit import SEARCH_LIMIT, limiter
from mimeme.media import Urls
from mimeme.search.rows import SqlRows

router = APIRouter(prefix="/search", tags=["Search"], responses=error_responses(403, 429, 500))


def get_client(request: Request) -> search.Client:
    return request.app.state.env.search


def get_rows(request: Request) -> search.Rows:
    return SqlRows(request.app.state.env.db)


def get_media_urls(request: Request) -> Urls:
    return request.app.state.env.media_urls


ClientDep = Annotated[search.Client, Depends(get_client)]
RowsDep = Annotated[search.Rows, Depends(get_rows)]
UrlsDep = Annotated[Urls, Depends(get_media_urls)]


@router.get("", response_model=search.Page, responses=error_responses(400, 503))
@limiter.limit(SEARCH_LIMIT)
async def search_text(
    request: Request,
    _auth: ReadonlyRequired,
    client: ClientDep,
    rows: RowsDep,
    media_urls: UrlsDep,
    q: Annotated[str, HttpQuery(min_length=1, max_length=200, description="Search query")],
    limit: Annotated[int, HttpQuery(ge=1, le=100)] = 20,
    offset: Annotated[int, HttpQuery(ge=0)] = 0,
    mode: Annotated[Literal["hybrid"] | None, HttpQuery()] = None,
) -> search.Page:
    query = search.Query(
        text=q,
        mode="hybrid" if mode == "hybrid" else "image",
        limit=limit,
        offset=offset,
    )
    started = time.monotonic()
    try:
        result = await search.run(query, client=client, rows=rows, media_urls=media_urls)
    except search.Error as exc:
        _log_search_failure(query, started, exc)
        raise _http(exc) from exc
    _log_search_completed(query, started, result)
    return result


@router.get("/similar/{image_id}", response_model=search.Page, responses=error_responses(404, 503))
@limiter.limit(SEARCH_LIMIT)
async def search_similar(
    request: Request,
    _auth: ReadonlyRequired,
    client: ClientDep,
    rows: RowsDep,
    media_urls: UrlsDep,
    image_id: int,
    limit: Annotated[int, HttpQuery(ge=1, le=100)] = 20,
) -> search.Page:
    query = search.Query(similar_image_id=image_id, limit=limit)
    started = time.monotonic()
    try:
        result = await search.run(query, client=client, rows=rows, media_urls=media_urls)
    except search.Error as exc:
        _log_search_failure(query, started, exc)
        raise _http(exc) from exc
    _log_search_completed(query, started, result)
    return result


def _log_search_completed(query: search.Query, started: float, result: search.Page) -> None:
    structlog.get_logger().info(
        "search_completed",
        mode=query.mode,
        query_length=len(query.text) if query.text is not None else None,
        result_count=len(result.results),
        matched_count=result.total,
        zero_results=result.total == 0,
        limit=query.limit,
        offset=query.offset,
        duration_ms=round((time.monotonic() - started) * 1000, 2),
        search_time_ms=result.search_time_ms,
        index_version=result.index_version,
    )


def _log_search_failure(query: search.Query, started: float, exc: search.Error) -> None:
    status_code = _http(exc).status_code
    structlog.get_logger().warning(
        "search_failed",
        mode=query.mode,
        query_length=len(query.text) if query.text is not None else None,
        limit=query.limit,
        offset=query.offset,
        status_code=status_code,
        error_type=type(exc).__name__,
        duration_ms=round((time.monotonic() - started) * 1000, 2),
        error=str(exc),
    )


def _http(exc: search.Error) -> HTTPException:
    if isinstance(exc, (search.Unavailable, search.Loading, search.Incompatible, search.Stale)):
        status_code = 503
    elif isinstance(exc, search.NotFound):
        status_code = 404
    elif isinstance(exc, search.Invalid):
        status_code = 400
    else:
        status_code = 500
    message = "Search failed" if status_code == 500 else str(exc)
    return HTTPException(status_code=status_code, detail=message)
