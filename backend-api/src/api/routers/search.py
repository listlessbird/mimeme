import asyncio
import time
from typing import Annotated, Literal

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy.orm import Session

from api.auth import ReadonlyRequired
from api.deps import IndexManagerDep
from api.models.search import SearchResponse
from api.rate_limit import SEARCH_LIMIT, limiter
from api.services.search import SearchService
from api.services.text_encoder import SearchTextEncoder
from shared.db import session_scope
from shared.models import IndexBuild

router = APIRouter(prefix="/search", tags=["Search"])
log = structlog.get_logger()

_last_index_check: float = 0.0
_INDEX_CHECK_INTERVAL: float = 60.0


def _ensure_index_loaded(db: Session, index_manager: IndexManagerDep) -> None:
    active_build = db.query(IndexBuild).filter(IndexBuild.is_active).first()
    if active_build is None:
        raise HTTPException(status_code=503, detail="Search index not loaded")

    if (not index_manager.is_loaded) or (index_manager.active_version != active_build.version):
        try:
            index_manager.load_active_index(db)
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="Search index not loaded")
        except Exception as exc:
            log.exception("search_index_reload_failed")
            raise HTTPException(status_code=500, detail=f"Failed to load search index: {exc}")


def _ensure_index_loaded_for_thread(index_manager: IndexManagerDep) -> None:
    global _last_index_check
    now = time.monotonic()
    if index_manager.is_loaded and (now - _last_index_check) < _INDEX_CHECK_INTERVAL:
        return
    with session_scope() as db:
        _ensure_index_loaded(db, index_manager)
    _last_index_check = now


def _search_by_embedding_for_thread(
    index_manager: IndexManagerDep,
    embedding: list[float],
    limit: int,
    mode: Literal["image", "text", "hybrid"] = "hybrid",
) -> list:
    with session_scope() as db:
        search_service = SearchService(index_manager)
        return search_service.search_by_embedding(
            embedding=embedding,
            limit=limit,
            db=db,
            mode=mode,
        )


def _find_similar_for_thread(
    index_manager: IndexManagerDep,
    image_id: int,
    limit: int,
) -> list:
    with session_scope() as db:
        search_service = SearchService(index_manager)
        return search_service.find_similar(
            image_id=image_id,
            limit=limit,
            db=db,
        )


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
        Literal["image", "text", "hybrid"],
        Query(description="Search mode: image (visual), text (caption/OCR), hybrid (both via RRF)"),
    ] = "hybrid",
) -> SearchResponse:
    start_time = time.perf_counter()

    await asyncio.to_thread(_ensure_index_loaded_for_thread, index_manager)

    # Encode query locally on CPU — no Temporal/Modal round-trip
    try:
        encoder = SearchTextEncoder.get_instance()
        embedding_arr = await asyncio.to_thread(encoder.encode, q)
        embedding = embedding_arr.tolist()
    except Exception as e:
        log.exception("search_query_encoding_failed", query=q, mode=mode)
        raise HTTPException(status_code=500, detail=f"Failed to encode query: {e}")

    try:
        results = await asyncio.to_thread(
            _search_by_embedding_for_thread,
            index_manager,
            embedding,
            limit + offset,
            mode,
        )
        paginated = results[offset : offset + limit]
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SearchResponse(
            query=q,
            results=paginated,
            total=len(results),
            limit=limit,
            offset=offset,
            search_time_ms=round(elapsed_ms, 2),
            index_version=index_manager.active_version,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        log.exception(
            "search_query_failed",
            query=q,
            mode=mode,
            limit=limit,
            offset=offset,
        )
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
    start_time = time.perf_counter()

    await asyncio.to_thread(_ensure_index_loaded_for_thread, index_manager)

    try:
        results = await asyncio.to_thread(
            _find_similar_for_thread,
            index_manager,
            image_id,
            limit + 1,
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
            index_version=index_manager.active_version,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
