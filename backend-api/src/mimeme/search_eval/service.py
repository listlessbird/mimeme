from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from mimeme import release, search
from mimeme.db import Db
from mimeme.db.schema import (
    Annotation,
    SearchEvalBaseline,
    SearchEvalJudgment,
    SearchEvalPoolCandidate,
    SearchEvalQuery,
    SearchEvalQueryExecution,
    SearchEvalResult,
    SearchEvalRun,
    SearchEvalScore,
    SearchEvalSnapshot,
)
from mimeme.db.schema import Image as ImageRow
from mimeme.media import Urls
from mimeme.search.rows import SqlRows
from mimeme.search_eval.metrics import Metrics, calculate
from mimeme.search_eval.model import (
    CandidateView,
    Comparison,
    Intent,
    JudgmentSave,
    JudgmentWorkspace,
    Overview,
    PoolResult,
    PreparedRun,
    QueryComparison,
    QuerySource,
    QuerySpec,
    QueryView,
    RankedImage,
    RunMetricsView,
    RunMode,
    RunView,
    WorkflowResult,
)


class NotFound(Exception):
    pass


class Conflict(Exception):
    pass


class Incomplete(Exception):
    pass


def _metric_view(metrics: Metrics | dict | None) -> RunMetricsView | None:
    if metrics is None:
        return None
    parsed = metrics if isinstance(metrics, Metrics) else Metrics.model_validate(metrics)
    return RunMetricsView.model_validate(parsed.model_dump(exclude={"per_query"}))


async def _query_views(db: Db) -> list[QueryView]:
    async with db.read_session() as session:
        pool_count = (
            select(func.count(SearchEvalPoolCandidate.image_id))
            .where(SearchEvalPoolCandidate.query_id == SearchEvalQuery.id)
            .correlate(SearchEvalQuery)
            .scalar_subquery()
        )
        judgment_count = (
            select(func.count(SearchEvalJudgment.image_id))
            .where(SearchEvalJudgment.query_id == SearchEvalQuery.id)
            .correlate(SearchEvalQuery)
            .scalar_subquery()
        )
        relevant_count = (
            select(func.count(SearchEvalJudgment.image_id))
            .where(
                SearchEvalJudgment.query_id == SearchEvalQuery.id,
                SearchEvalJudgment.grade >= 2,
            )
            .correlate(SearchEvalQuery)
            .scalar_subquery()
        )
        rows = (
            await session.execute(
                select(
                    SearchEvalQuery,
                    pool_count.label("candidate_count"),
                    judgment_count.label("judgment_count"),
                    relevant_count.label("relevant_count"),
                ).order_by(SearchEvalQuery.status, SearchEvalQuery.id)
            )
        ).all()
    return [
        QueryView(
            id=row.id,
            text=row.text,
            intent=row.intent,
            source=row.source,
            status=row.status,
            candidate_count=candidate_count,
            judgment_count=judgment_count,
            relevant_count=relevant_count,
            created_at=row.created_at,
        )
        for row, candidate_count, judgment_count, relevant_count in rows
    ]


async def _missing_judgments(db: Db, run_id: str) -> int:
    async with db.read_session() as session:
        count = await session.scalar(
            select(func.count())
            .select_from(SearchEvalResult)
            .outerjoin(
                SearchEvalJudgment,
                (SearchEvalJudgment.query_id == SearchEvalResult.query_id)
                & (SearchEvalJudgment.image_id == SearchEvalResult.image_id),
            )
            .where(SearchEvalResult.run_id == run_id, SearchEvalJudgment.image_id.is_(None))
        )
    return count or 0


async def _run_view(db: Db, row: SearchEvalRun) -> RunView:
    if row.mode not in ("image", "hybrid"):
        raise RuntimeError(f"Invalid search eval mode: {row.mode}")
    if row.status not in (
        "queued",
        "running",
        "needs_judgments",
        "complete",
        "failed",
        "cancelled",
    ):
        raise RuntimeError(f"Invalid search eval status: {row.status}")
    if row.phase not in (None, "preparing", "searching", "calculating_metrics", "finalizing"):
        raise RuntimeError(f"Invalid search eval phase: {row.phase}")
    score = None
    if row.score_snapshot_id is not None:
        async with db.read_session() as session:
            score = await session.get(SearchEvalScore, (row.id, row.score_snapshot_id))
    return RunView(
        id=row.id,
        mode=row.mode,
        status=row.status,
        phase=row.phase,
        progress_completed=row.progress_completed,
        progress_total=row.progress_total,
        index_version=row.index_version,
        release_id=row.release_id,
        snapshot_id=row.snapshot_id,
        metrics=_metric_view(score.metrics if score else None),
        missing_judgments=await _missing_judgments(db, row.id),
        error=row.error,
        created_at=row.created_at,
        started_at=row.started_at,
        completed_at=row.completed_at,
    )


async def get_overview(db: Db) -> Overview:
    queries = await _query_views(db)
    async with db.read_session() as session:
        run_rows = (
            await session.scalars(
                select(SearchEvalRun).order_by(SearchEvalRun.created_at.desc()).limit(20)
            )
        ).all()
        baseline = await session.get(SearchEvalBaseline, 1)
    runs = [await _run_view(db, row) for row in run_rows]
    active = [query for query in queries if query.status == "active"]
    candidates = sum(query.candidate_count for query in active)
    judgments = sum(query.judgment_count for query in active)
    return Overview(
        queries=queries,
        recent_runs=runs,
        baseline_run_id=baseline.run_id if baseline else None,
        active_query_count=len(active),
        candidate_count=candidates,
        judgment_count=judgments,
        unjudged_count=max(0, candidates - judgments),
    )


async def create_query(db: Db, *, text: str, intent: Intent, source: QuerySource) -> QueryView:
    normalized = " ".join(text.split())
    if not normalized:
        raise Incomplete("Query text cannot be empty")
    row = SearchEvalQuery(text=normalized, intent=intent, source=source, status="active")
    try:
        async with db.write_session() as session:
            session.add(row)
            await session.flush()
    except IntegrityError as exc:
        raise Conflict("That query already exists") from exc
    views = await _query_views(db)
    return next(view for view in views if view.id == row.id)


async def disable_query(db: Db, query_id: int) -> None:
    async with db.write_session() as session:
        row = await session.get(SearchEvalQuery, query_id)
        if row is None:
            raise NotFound("Query not found")
        row.status = "disabled"


async def pool_query(
    db: Db,
    query_id: int,
    *,
    client: search.Client,
    media_urls: Urls,
) -> PoolResult:
    async with db.read_session() as session:
        query_row = await session.get(SearchEvalQuery, query_id)
    if query_row is None:
        raise NotFound("Query not found")

    pages: dict[RunMode, search.Page] = {}
    for mode in ("image", "hybrid"):
        pages[mode] = await search.run(
            search.Query(text=query_row.text, mode=mode, limit=20),
            client=client,
            rows=SqlRows(db),
            media_urls=media_urls,
        )
    versions = {page.index_version for page in pages.values()}
    if len(versions) != 1 or None in versions:
        raise Conflict("The active search index changed while building the pool")
    version = next(iter(versions))
    assert version is not None

    ranks = {
        mode: {result.id: rank for rank, result in enumerate(page.results, 1)}
        for mode, page in pages.items()
    }
    image_ids = set(ranks["image"]) | set(ranks["hybrid"])
    async with db.write_session() as session:
        existing = {
            row.image_id: row
            for row in (
                await session.scalars(
                    select(SearchEvalPoolCandidate).where(
                        SearchEvalPoolCandidate.query_id == query_id
                    )
                )
            ).all()
        }
        added = 0
        for image_id in image_ids:
            row = existing.get(image_id)
            if row is None:
                row = SearchEvalPoolCandidate(query_id=query_id, image_id=image_id)
                session.add(row)
                added += 1
            row.image_rank = ranks["image"].get(image_id)
            row.hybrid_rank = ranks["hybrid"].get(image_id)
        await session.flush()
        count = await session.scalar(
            select(func.count())
            .select_from(SearchEvalPoolCandidate)
            .where(SearchEvalPoolCandidate.query_id == query_id)
        )
    return PoolResult(
        query_id=query_id,
        candidate_count=count or 0,
        added_count=added,
        index_version=version,
    )


async def add_candidate(db: Db, query_id: int, image_id: int) -> None:
    async with db.write_session() as session:
        query_row = await session.get(SearchEvalQuery, query_id)
        image_row = await session.get(ImageRow, image_id)
        if query_row is None:
            raise NotFound("Query not found")
        if image_row is None:
            raise NotFound("Image not found")
        row = await session.get(SearchEvalPoolCandidate, (query_id, image_id))
        if row is None:
            session.add(SearchEvalPoolCandidate(query_id=query_id, image_id=image_id, manual=True))
        else:
            row.manual = True


async def get_judgment_workspace(
    db: Db, media_urls: Urls, *, query_id: int | None = None
) -> JudgmentWorkspace:
    query_views = [query for query in await _query_views(db) if query.status == "active"]
    if not query_views:
        raise NotFound("No active queries")
    selected = next((query for query in query_views if query.id == query_id), None)
    if selected is None and query_id is not None:
        raise NotFound("Query not found")
    if selected is None:
        selected = next(
            (query for query in query_views if query.judgment_count < query.candidate_count),
            query_views[0],
        )
    selected_index = query_views.index(selected)

    async with db.read_session() as session:
        rows = (
            await session.execute(
                select(SearchEvalPoolCandidate, ImageRow, Annotation, SearchEvalJudgment)
                .join(ImageRow, ImageRow.id == SearchEvalPoolCandidate.image_id)
                .outerjoin(Annotation, Annotation.image_id == ImageRow.id)
                .outerjoin(
                    SearchEvalJudgment,
                    (SearchEvalJudgment.query_id == SearchEvalPoolCandidate.query_id)
                    & (SearchEvalJudgment.image_id == SearchEvalPoolCandidate.image_id),
                )
                .where(SearchEvalPoolCandidate.query_id == selected.id)
                .order_by(SearchEvalJudgment.grade.is_not(None), ImageRow.id)
            )
        ).all()
    candidates = [
        CandidateView(
            image_id=image.id,
            url=media_urls.resolve(image.s3_key) if image.s3_key else None,
            caption=annotation.caption_text if annotation else None,
            ocr_text=annotation.ocr_text if annotation else None,
            width=image.width,
            height=image.height,
            grade=judgment.grade if judgment else None,
            revision=judgment.revision if judgment else 0,
        )
        for _, image, annotation, judgment in rows
    ]
    return JudgmentWorkspace(
        query=selected,
        candidates=candidates,
        previous_query_id=query_views[selected_index - 1].id if selected_index > 0 else None,
        next_query_id=(
            query_views[selected_index + 1].id if selected_index + 1 < len(query_views) else None
        ),
    )


async def save_judgment(
    db: Db,
    *,
    query_id: int,
    image_id: int,
    grade: int,
    revision: int,
) -> JudgmentSave:
    async with db.write_session() as session:
        pool_row = await session.get(SearchEvalPoolCandidate, (query_id, image_id))
        if pool_row is None:
            raise NotFound("Candidate is not in this query's judgment pool")
        row = await session.get(SearchEvalJudgment, (query_id, image_id))
        if row is None:
            if revision != 0:
                raise Conflict("Judgment changed in another tab")
            row = SearchEvalJudgment(query_id=query_id, image_id=image_id, grade=grade, revision=1)
            session.add(row)
        else:
            if row.revision != revision:
                raise Conflict("Judgment changed in another tab")
            row.grade = grade
            row.revision += 1
        await session.flush()
        saved_revision = row.revision
    return JudgmentSave(
        query_id=query_id,
        image_id=image_id,
        grade=grade,
        revision=saved_revision,
    )


async def clear_judgment(
    db: Db,
    *,
    query_id: int,
    image_id: int,
    revision: int,
) -> None:
    async with db.write_session() as session:
        row = await session.get(SearchEvalJudgment, (query_id, image_id))
        if row is None:
            return
        if row.revision != revision:
            raise Conflict("Judgment changed in another tab")
        await session.delete(row)


async def _snapshot(
    db: Db,
    query_ids: Iterable[int] | None = None,
    *,
    require_complete_pool: bool = True,
) -> SearchEvalSnapshot:
    async with db.read_session() as session:
        query_statement = select(SearchEvalQuery)
        if query_ids is None:
            query_statement = query_statement.where(SearchEvalQuery.status == "active")
        else:
            query_statement = query_statement.where(SearchEvalQuery.id.in_(list(query_ids)))
        queries = (await session.scalars(query_statement.order_by(SearchEvalQuery.id))).all()
        if not queries:
            raise Incomplete("Add at least one active query before running an eval")
        ids = [query.id for query in queries]
        judgments = (
            await session.scalars(
                select(SearchEvalJudgment)
                .where(SearchEvalJudgment.query_id.in_(ids))
                .order_by(SearchEvalJudgment.query_id, SearchEvalJudgment.image_id)
            )
        ).all()
        pool_count_rows = await session.execute(
            select(
                SearchEvalPoolCandidate.query_id,
                func.count(SearchEvalPoolCandidate.image_id),
            )
            .where(SearchEvalPoolCandidate.query_id.in_(ids))
            .group_by(SearchEvalPoolCandidate.query_id)
        )
        pool_counts: dict[int, int] = {
            query_id: count for query_id, count in pool_count_rows.tuples()
        }

    by_query: dict[int, list[dict[str, int]]] = {query_id: [] for query_id in ids}
    for judgment in judgments:
        by_query[judgment.query_id].append({"image_id": judgment.image_id, "grade": judgment.grade})
    for query in queries:
        judged = by_query[query.id]
        if require_complete_pool and pool_counts.get(query.id, 0) == 0:
            raise Incomplete(f'Pool results before running "{query.text}"')
        if require_complete_pool and len(judged) < pool_counts.get(query.id, 0):
            raise Incomplete(f'Finish judging "{query.text}" before running the eval')
        if not any(row["grade"] >= 2 for row in judged):
            raise Incomplete(f'Add a relevant result or disable "{query.text}"')

    query_payload = [
        {
            "id": query.id,
            "text": query.text,
            "intent": query.intent,
            "source": query.source,
        }
        for query in queries
    ]
    payload = {"queries": query_payload, "judgments": by_query}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    query_encoded = json.dumps(query_payload, sort_keys=True, separators=(",", ":")).encode()
    content_hash = hashlib.sha256(encoded).hexdigest()
    query_hash = hashlib.sha256(query_encoded).hexdigest()

    async with db.write_session() as session:
        existing = await session.scalar(
            select(SearchEvalSnapshot).where(SearchEvalSnapshot.content_hash == content_hash)
        )
        if existing is not None:
            return existing
        row = SearchEvalSnapshot(
            id=uuid.uuid4().hex,
            content_hash=content_hash,
            query_hash=query_hash,
            query_count=len(queries),
            judgment_count=len(judgments),
            payload=payload,
        )
        session.add(row)
        await session.flush()
        return row


def _snapshot_judgments(snapshot: SearchEvalSnapshot) -> dict[int, dict[int, int]]:
    raw = snapshot.payload["judgments"]
    return {
        int(query_id): {int(row["image_id"]): int(row["grade"]) for row in rows}
        for query_id, rows in raw.items()
    }


async def _run_data(
    db: Db, run_id: str
) -> tuple[dict[int, list[int]], dict[int, float], list[SearchEvalResult]]:
    async with db.read_session() as session:
        rows = (
            await session.scalars(
                select(SearchEvalResult)
                .where(SearchEvalResult.run_id == run_id)
                .order_by(SearchEvalResult.query_id, SearchEvalResult.rank)
            )
        ).all()
        executions = (
            await session.scalars(
                select(SearchEvalQueryExecution).where(SearchEvalQueryExecution.run_id == run_id)
            )
        ).all()
    rankings: dict[int, list[int]] = {}
    for row in rows:
        rankings.setdefault(row.query_id, []).append(row.image_id)
    latencies = {row.query_id: row.search_time_ms for row in executions}
    return rankings, latencies, list(rows)


async def get_run(db: Db, run_id: str) -> RunView:
    async with db.read_session() as session:
        row = await session.get(SearchEvalRun, run_id)
    if row is None:
        raise NotFound("Run not found")
    return await _run_view(db, row)


async def create_run(db: Db, *, run_id: str, mode: RunMode, workflow_id: str) -> RunView:
    async with db.read_session() as session:
        active = await session.scalar(
            select(func.count())
            .select_from(SearchEvalRun)
            .where(SearchEvalRun.status.in_(("queued", "running")))
        )
    if active:
        raise Conflict("Another search evaluation is already running")
    snapshot = await _snapshot(db)
    row = SearchEvalRun(
        id=run_id,
        snapshot_id=snapshot.id,
        mode=mode,
        status="queued",
        phase="preparing",
        workflow_id=workflow_id,
        progress_completed=0,
        progress_total=snapshot.query_count,
        release_id=release.ID,
    )
    async with db.write_session() as session:
        session.add(row)
        await session.flush()
    return await _run_view(db, row)


async def queue_rescore(db: Db, run_id: str, *, workflow_id: str) -> RunView:
    if await _missing_judgments(db, run_id):
        raise Incomplete("Finish the missing judgments before rescoring")
    async with db.write_session() as session:
        row = await session.get(SearchEvalRun, run_id)
        if row is None:
            raise NotFound("Run not found")
        if row.status != "needs_judgments":
            raise Incomplete("Only a run awaiting judgments can be rescored")
        row.status = "queued"
        row.phase = "calculating_metrics"
        row.workflow_id = workflow_id
        row.error = None
    return await get_run(db, run_id)


async def prepare_run(db: Db, run_id: str, *, index_version: str) -> PreparedRun:
    async with db.write_session() as session:
        row = await session.get(SearchEvalRun, run_id)
        if row is None:
            raise NotFound("Run not found")
        if row.status not in ("queued", "running"):
            raise Conflict(f"Run cannot start from status {row.status}")
        if row.index_version is not None and row.index_version != index_version:
            raise Conflict("The search index changed before the run resumed")
        snapshot = await session.get(SearchEvalSnapshot, row.snapshot_id)
        assert snapshot is not None
        if row.mode not in ("image", "hybrid"):
            raise RuntimeError(f"Invalid search eval mode: {row.mode}")
        row.status = "running"
        row.phase = "searching"
        row.index_version = index_version
        row.started_at = row.started_at or datetime.now(UTC)
        queries = [
            QuerySpec(id=int(item["id"]), text=str(item["text"]))
            for item in snapshot.payload["queries"]
        ]
        return PreparedRun(
            run_id=row.id,
            mode=row.mode,
            index_version=index_version,
            queries=queries,
        )


async def record_query(
    db: Db,
    *,
    run_id: str,
    query_id: int,
    mode: RunMode,
    expected_index_version: str,
    page: search.Page,
) -> bool:
    if page.index_version != expected_index_version:
        raise Conflict("The active search index changed during the eval run")
    async with db.write_session() as session:
        existing = await session.get(SearchEvalQueryExecution, (run_id, query_id))
        if existing is not None:
            return False
        row = await session.get(SearchEvalRun, run_id)
        if row is None:
            raise NotFound("Run not found")
        if row.index_version != expected_index_version:
            raise Conflict("The eval run index version does not match retrieval")

        session.add(
            SearchEvalQueryExecution(
                run_id=run_id,
                query_id=query_id,
                result_count=len(page.results),
                search_time_ms=page.search_time_ms,
            )
        )
        for rank, result in enumerate(page.results, 1):
            session.add(
                SearchEvalResult(
                    run_id=run_id,
                    query_id=query_id,
                    rank=rank,
                    image_id=result.id,
                    score=result.score,
                )
            )
            candidate = await session.get(SearchEvalPoolCandidate, (query_id, result.id))
            if candidate is None:
                candidate = SearchEvalPoolCandidate(query_id=query_id, image_id=result.id)
                session.add(candidate)
            if mode == "image":
                candidate.image_rank = rank
            else:
                candidate.hybrid_rank = rank
        completed = await session.scalar(
            select(func.count())
            .select_from(SearchEvalQueryExecution)
            .where(SearchEvalQueryExecution.run_id == run_id)
        )
        row.progress_completed = min(row.progress_total, (completed or 0) + 1)
    return True


async def query_recorded(db: Db, run_id: str, query_id: int) -> bool:
    async with db.read_session() as session:
        return await session.get(SearchEvalQueryExecution, (run_id, query_id)) is not None


async def _score_one(db: Db, run_id: str, snapshot: SearchEvalSnapshot) -> Metrics:
    rankings, latencies, _ = await _run_data(db, run_id)
    metrics = await asyncio.to_thread(calculate, rankings, _snapshot_judgments(snapshot), latencies)
    async with db.write_session() as session:
        score = await session.get(SearchEvalScore, (run_id, snapshot.id))
        if score is None:
            session.add(
                SearchEvalScore(
                    run_id=run_id,
                    snapshot_id=snapshot.id,
                    metrics=metrics.model_dump(mode="json"),
                )
            )
        row = await session.get(SearchEvalRun, run_id)
        assert row is not None
        row.score_snapshot_id = snapshot.id
    return metrics


async def score_run(db: Db, run_id: str) -> WorkflowResult:
    async with db.read_session() as session:
        run = await session.get(SearchEvalRun, run_id)
        if run is None:
            raise NotFound("Run not found")
        source_snapshot = await session.get(SearchEvalSnapshot, run.snapshot_id)
        assert source_snapshot is not None
    query_ids = [int(query["id"]) for query in source_snapshot.payload["queries"]]
    rankings, latencies, _ = await _run_data(db, run_id)
    if set(latencies) != set(query_ids):
        raise Incomplete("The eval run has not retrieved every query")

    async with db.read_session() as session:
        judged_rows = await session.execute(
            select(SearchEvalJudgment.query_id, SearchEvalJudgment.image_id).where(
                SearchEvalJudgment.query_id.in_(query_ids)
            )
        )
        judged_pairs = set(judged_rows.tuples())
    missing = sum(
        (query_id, image_id) not in judged_pairs
        for query_id, image_ids in rankings.items()
        for image_id in image_ids
    )
    if missing:
        async with db.write_session() as session:
            row = await session.get(SearchEvalRun, run_id)
            assert row is not None
            row.status = "needs_judgments"
            row.phase = None
        return WorkflowResult(run_id=run_id, status="needs_judgments")

    async with db.write_session() as session:
        row = await session.get(SearchEvalRun, run_id)
        assert row is not None
        row.status = "running"
        row.phase = "calculating_metrics"

    snapshot = await _snapshot(db, query_ids, require_complete_pool=False)
    await _score_one(db, run_id, snapshot)

    async with db.read_session() as session:
        baseline = await session.get(SearchEvalBaseline, 1)
        baseline_run = await session.get(SearchEvalRun, baseline.run_id) if baseline else None
        baseline_snapshot = (
            await session.get(SearchEvalSnapshot, baseline_run.snapshot_id)
            if baseline_run is not None
            else None
        )
    if (
        baseline_run is not None
        and baseline_run.id != run_id
        and baseline_snapshot is not None
        and baseline_snapshot.query_hash == snapshot.query_hash
    ):
        await _score_one(db, baseline_run.id, snapshot)

    async with db.write_session() as session:
        row = await session.get(SearchEvalRun, run_id)
        assert row is not None
        row.phase = "finalizing"

    async with db.write_session() as session:
        row = await session.get(SearchEvalRun, run_id)
        assert row is not None
        row.status = "complete"
        row.phase = None
        row.completed_at = datetime.now(UTC)
    return WorkflowResult(run_id=run_id, status="complete")


async def fail_run(db: Db, run_id: str, error: str) -> WorkflowResult:
    async with db.write_session() as session:
        row = await session.get(SearchEvalRun, run_id)
        if row is None:
            raise NotFound("Run not found")
        if row.status not in ("complete", "needs_judgments", "cancelled"):
            row.status = "failed"
            row.phase = None
            row.error = error[:2000]
            row.completed_at = datetime.now(UTC)
    return WorkflowResult(run_id=run_id, status="failed")


async def set_baseline(db: Db, run_id: str) -> RunView:
    async with db.write_session() as session:
        run = await session.get(SearchEvalRun, run_id)
        if run is None:
            raise NotFound("Run not found")
        if run.status != "complete":
            raise Incomplete("Only a complete run can become the baseline")
        baseline = await session.get(SearchEvalBaseline, 1)
        if baseline is None:
            session.add(SearchEvalBaseline(id=1, run_id=run_id))
        else:
            baseline.run_id = run_id
    return await _run_view(db, run)


async def compare_runs(
    db: Db,
    media_urls: Urls,
    *,
    baseline_run_id: str,
    candidate_run_id: str,
) -> Comparison:
    async with db.read_session() as session:
        baseline = await session.get(SearchEvalRun, baseline_run_id)
        candidate = await session.get(SearchEvalRun, candidate_run_id)
        if baseline is None or candidate is None:
            raise NotFound("Run not found")
        if baseline.status != "complete" or candidate.status != "complete":
            raise Incomplete("Both runs must be complete before comparison")
        if (
            baseline.score_snapshot_id is None
            or candidate.score_snapshot_id is None
            or baseline.score_snapshot_id != candidate.score_snapshot_id
        ):
            raise Conflict("Runs must be scored against the same judgment snapshot")
        snapshot = await session.get(SearchEvalSnapshot, candidate.score_snapshot_id)
        baseline_score = await session.get(
            SearchEvalScore, (baseline.id, baseline.score_snapshot_id)
        )
        candidate_score = await session.get(
            SearchEvalScore, (candidate.id, candidate.score_snapshot_id)
        )
        assert snapshot is not None and baseline_score is not None and candidate_score is not None

    judgments = _snapshot_judgments(snapshot)
    baseline_metrics = Metrics.model_validate(baseline_score.metrics)
    candidate_metrics = Metrics.model_validate(candidate_score.metrics)
    baseline_rankings, _, _ = await _run_data(db, baseline.id)
    candidate_rankings, _, _ = await _run_data(db, candidate.id)
    baseline_per_query = {row.query_id: row for row in baseline_metrics.per_query}
    candidate_per_query = {row.query_id: row for row in candidate_metrics.per_query}

    all_image_ids = {
        image_id
        for rankings in (baseline_rankings, candidate_rankings)
        for image_ids in rankings.values()
        for image_id in image_ids
    }
    async with db.read_session() as session:
        image_rows = (
            await session.scalars(select(ImageRow).where(ImageRow.id.in_(all_image_ids)))
        ).all()
    images = {image.id: image for image in image_rows}
    query_payload = {int(query["id"]): query for query in snapshot.payload["queries"]}

    def image_url(image_id: int) -> str | None:
        image = images.get(image_id)
        if image is None or image.s3_key is None:
            return None
        return media_urls.resolve(image.s3_key)

    def ranked(query_id: int, rankings: dict[int, list[int]]) -> list[RankedImage]:
        return [
            RankedImage(
                image_id=image_id,
                rank=rank,
                url=image_url(image_id),
                grade=judgments.get(query_id, {}).get(image_id),
            )
            for rank, image_id in enumerate(rankings.get(query_id, []), 1)
        ]

    query_comparisons: list[QueryComparison] = []
    for query_id, query in query_payload.items():
        baseline_query = baseline_per_query[query_id]
        candidate_query = candidate_per_query[query_id]
        delta = candidate_query.ndcg_at_10 - baseline_query.ndcg_at_10
        query_comparisons.append(
            QueryComparison(
                query_id=query_id,
                text=query["text"],
                intent=query["intent"],
                baseline_ndcg_at_10=baseline_query.ndcg_at_10,
                candidate_ndcg_at_10=candidate_query.ndcg_at_10,
                delta_ndcg_at_10=delta,
                baseline_results=ranked(query_id, baseline_rankings),
                candidate_results=ranked(query_id, candidate_rankings),
            )
        )
    query_comparisons.sort(key=lambda query: query.delta_ndcg_at_10)
    epsilon = 1e-9
    return Comparison(
        baseline=(await _run_view(db, baseline)).model_copy(
            update={"metrics": _metric_view(baseline_metrics)}
        ),
        candidate=(await _run_view(db, candidate)).model_copy(
            update={"metrics": _metric_view(candidate_metrics)}
        ),
        delta_ndcg_at_10=candidate_metrics.ndcg_at_10 - baseline_metrics.ndcg_at_10,
        delta_success_at_5=candidate_metrics.success_at_5 - baseline_metrics.success_at_5,
        delta_mrr_at_10=candidate_metrics.mrr_at_10 - baseline_metrics.mrr_at_10,
        delta_latency_p95_ms=(candidate_metrics.latency_p95_ms - baseline_metrics.latency_p95_ms),
        improved_queries=sum(query.delta_ndcg_at_10 > epsilon for query in query_comparisons),
        unchanged_queries=sum(
            abs(query.delta_ndcg_at_10) <= epsilon for query in query_comparisons
        ),
        regressed_queries=sum(query.delta_ndcg_at_10 < -epsilon for query in query_comparisons),
        queries=query_comparisons,
    )
