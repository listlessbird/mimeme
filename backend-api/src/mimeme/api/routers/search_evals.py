from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from mimeme.api.auth import AdminRequired
from mimeme.api.deps import DbDep, EnvDep, SearchDep, UrlsDep
from mimeme.api.models.errors import error_responses
from mimeme.api.models.search_evals import (
    AddSearchEvalCandidateRequest,
    CreateSearchEvalQueryRequest,
    CreateSearchEvalRunRequest,
    SaveSearchEvalJudgmentRequest,
)
from mimeme.search_eval import model as eval_model
from mimeme.search_eval import service as search_eval
from mimeme.search_eval import submit as eval_submit

router = APIRouter(
    prefix="/search-evals",
    tags=["Search evaluations"],
    responses=error_responses(403, 429, 500),
)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, search_eval.NotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (search_eval.Conflict, search_eval.Incomplete)):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=500, detail="Search evaluation failed")


@router.get("", response_model=eval_model.Overview)
async def overview(_auth: AdminRequired, db: DbDep) -> eval_model.Overview:
    return await search_eval.get_overview(db)


@router.post("/queries", response_model=eval_model.QueryView, status_code=201)
async def create_query(
    _auth: AdminRequired,
    db: DbDep,
    body: CreateSearchEvalQueryRequest,
) -> eval_model.QueryView:
    try:
        return await search_eval.create_query(
            db, text=body.text, intent=body.intent, source=body.source
        )
    except (search_eval.Conflict, search_eval.Incomplete) as exc:
        raise _http_error(exc) from exc


@router.delete("/queries/{query_id}", status_code=204)
async def disable_query(_auth: AdminRequired, db: DbDep, query_id: int) -> None:
    try:
        await search_eval.disable_query(db, query_id)
    except search_eval.NotFound as exc:
        raise _http_error(exc) from exc


@router.post("/queries/{query_id}/pool", response_model=eval_model.PoolResult)
async def pool_query(
    _auth: AdminRequired,
    db: DbDep,
    search_client: SearchDep,
    media_urls: UrlsDep,
    query_id: int,
) -> eval_model.PoolResult:
    try:
        return await search_eval.pool_query(
            db, query_id, client=search_client, media_urls=media_urls
        )
    except (search_eval.NotFound, search_eval.Conflict) as exc:
        raise _http_error(exc) from exc


@router.post("/queries/{query_id}/candidates", status_code=204)
async def add_candidate(
    _auth: AdminRequired,
    db: DbDep,
    query_id: int,
    body: AddSearchEvalCandidateRequest,
) -> None:
    try:
        await search_eval.add_candidate(db, query_id, body.image_id)
    except search_eval.NotFound as exc:
        raise _http_error(exc) from exc


@router.get("/judgments", response_model=eval_model.JudgmentWorkspace)
async def judgment_workspace(
    _auth: AdminRequired,
    db: DbDep,
    media_urls: UrlsDep,
    query_id: Annotated[int | None, Query()] = None,
) -> eval_model.JudgmentWorkspace:
    try:
        return await search_eval.get_judgment_workspace(db, media_urls, query_id=query_id)
    except search_eval.NotFound as exc:
        raise _http_error(exc) from exc


@router.put(
    "/queries/{query_id}/judgments/{image_id}",
    response_model=eval_model.JudgmentSave,
)
async def save_judgment(
    _auth: AdminRequired,
    db: DbDep,
    query_id: int,
    image_id: int,
    body: SaveSearchEvalJudgmentRequest,
) -> eval_model.JudgmentSave:
    try:
        return await search_eval.save_judgment(
            db,
            query_id=query_id,
            image_id=image_id,
            grade=body.grade,
            revision=body.revision,
        )
    except (search_eval.NotFound, search_eval.Conflict) as exc:
        raise _http_error(exc) from exc


@router.delete("/queries/{query_id}/judgments/{image_id}", status_code=204)
async def clear_judgment(
    _auth: AdminRequired,
    db: DbDep,
    query_id: int,
    image_id: int,
    revision: Annotated[int, Query(ge=1)],
) -> None:
    try:
        await search_eval.clear_judgment(
            db,
            query_id=query_id,
            image_id=image_id,
            revision=revision,
        )
    except search_eval.Conflict as exc:
        raise _http_error(exc) from exc


@router.post("/runs", response_model=eval_model.RunView, status_code=202)
async def create_run(
    _auth: AdminRequired,
    env: EnvDep,
    body: CreateSearchEvalRunRequest,
) -> eval_model.RunView:
    try:
        return await eval_submit.submit_run(env, mode=body.mode)
    except (search_eval.Conflict, search_eval.Incomplete) as exc:
        raise _http_error(exc) from exc


@router.get("/runs/{run_id}", response_model=eval_model.RunView)
async def get_run(_auth: AdminRequired, db: DbDep, run_id: str) -> eval_model.RunView:
    try:
        return await search_eval.get_run(db, run_id)
    except search_eval.NotFound as exc:
        raise _http_error(exc) from exc


@router.post("/runs/{run_id}/finalize", response_model=eval_model.RunView, status_code=202)
async def finalize_run(_auth: AdminRequired, env: EnvDep, run_id: str) -> eval_model.RunView:
    try:
        return await eval_submit.submit_rescore(env, run_id)
    except (search_eval.NotFound, search_eval.Conflict, search_eval.Incomplete) as exc:
        raise _http_error(exc) from exc


@router.put("/runs/{run_id}/baseline", response_model=eval_model.RunView)
async def set_baseline(_auth: AdminRequired, db: DbDep, run_id: str) -> eval_model.RunView:
    try:
        return await search_eval.set_baseline(db, run_id)
    except (search_eval.NotFound, search_eval.Incomplete) as exc:
        raise _http_error(exc) from exc


@router.get("/compare", response_model=eval_model.Comparison)
async def compare(
    _auth: AdminRequired,
    db: DbDep,
    media_urls: UrlsDep,
    baseline_run_id: Annotated[str, Query(min_length=1, max_length=32)],
    candidate_run_id: Annotated[str, Query(min_length=1, max_length=32)],
) -> eval_model.Comparison:
    try:
        return await search_eval.compare_runs(
            db,
            media_urls,
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
        )
    except (search_eval.NotFound, search_eval.Conflict, search_eval.Incomplete) as exc:
        raise _http_error(exc) from exc
