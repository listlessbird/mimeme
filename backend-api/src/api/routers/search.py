from __future__ import annotations

import asyncio
import time
import uuid
from datetime import timedelta

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.orm import Session

from api.deps import DbSession, IndexManagerDep, TemporalClientDep
from api.models.search import SearchResponse
from api.services.search import SearchService
from shared.config import settings
from shared.models import IndexBuild
from workflows import EncodeQueryWorkflow

router = APIRouter()


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
            raise HTTPException(status_code=500, detail=f"Failed to load search index: {exc}")


@router.get("", response_model=SearchResponse)
async def search(
    db: DbSession,
    index_manager: IndexManagerDep,
    temporal: TemporalClientDep,
    q: str = Query(..., min_length=1, max_length=200, description="Search query"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> SearchResponse:
    start_time = time.perf_counter()

    await asyncio.to_thread(_ensure_index_loaded, db, index_manager)

    try:
        workflow_id = f"search-{uuid.uuid4().hex[:12]}"
        encode_result = await temporal.execute_workflow(
            EncodeQueryWorkflow.run,
            q,
            id=workflow_id,
            task_queue=settings.temporal_task_queue_cpu,
            execution_timeout=timedelta(seconds=60),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to encode query: {e}")

    search_service = SearchService(index_manager)

    try:
        results = await asyncio.to_thread(
            search_service.search_by_embedding,
            embedding=encode_result.embedding,
            limit=limit + offset,
            db=db,
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
        raise HTTPException(status_code=500, detail="Search failed")


@router.get("/similar/{image_id}", response_model=SearchResponse)
async def find_similar(
    image_id: int,
    db: DbSession,
    index_manager: IndexManagerDep,
    limit: int = Query(default=20, ge=1, le=100),
) -> SearchResponse:
    start_time = time.perf_counter()

    await asyncio.to_thread(_ensure_index_loaded, db, index_manager)

    search_service = SearchService(index_manager)

    try:
        results = await asyncio.to_thread(
            search_service.find_similar,
            image_id=image_id,
            limit=limit + 1,
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
            index_version=index_manager.active_version,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
