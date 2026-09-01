from __future__ import annotations

from typing import Protocol

from temporalio import activity
from temporalio.exceptions import ApplicationError

from mimeme import search
from mimeme.db import Db
from mimeme.media import Urls
from mimeme.search.rows import SqlRows
from mimeme.search_eval import rule
from mimeme.search_eval import service as evals
from mimeme.search_eval.model import (
    FailureInput,
    PreparedRun,
    RetrievalBatch,
    WorkflowInput,
    WorkflowResult,
)


class Deps(Protocol):
    db: Db
    search: search.Client
    media_urls: Urls


class Activities:
    def __init__(self, env: Deps) -> None:
        self._env = env

    @activity.defn(name=rule.PREPARE_ACTIVITY)
    async def prepare(self, input: WorkflowInput) -> PreparedRun:
        try:
            status = await self._env.search.status()
            if not status.ready or status.serving_version is None:
                raise evals.Conflict("Search has no active index generation")
            return await evals.prepare_run(
                self._env.db,
                input.run_id,
                index_version=status.serving_version,
            )
        except (evals.Conflict, evals.Incomplete, evals.NotFound) as exc:
            raise ApplicationError(str(exc), non_retryable=True) from exc

    @activity.defn(name=rule.RETRIEVE_ACTIVITY)
    async def retrieve(self, input: RetrievalBatch) -> None:
        for query in input.queries:
            if await evals.query_recorded(self._env.db, input.run_id, query.id):
                activity.heartbeat({"query_id": query.id, "outcome": "already_recorded"})
                continue
            page = await search.run(
                search.Query(text=query.text, recipe_id=input.recipe_id, limit=10),
                client=self._env.search,
                rows=SqlRows(self._env.db),
                media_urls=self._env.media_urls,
            )
            try:
                await evals.record_query(
                    self._env.db,
                    run_id=input.run_id,
                    query_id=query.id,
                    recipe_id=input.recipe_id,
                    expected_index_version=input.index_version,
                    page=page,
                )
            except (evals.Conflict, evals.Incomplete, evals.NotFound) as exc:
                raise ApplicationError(str(exc), non_retryable=True) from exc
            activity.heartbeat({"query_id": query.id, "outcome": "recorded"})

    @activity.defn(name=rule.SCORE_ACTIVITY)
    async def score(self, input: WorkflowInput) -> WorkflowResult:
        try:
            return await evals.score_run(self._env.db, input.run_id)
        except (evals.Conflict, evals.Incomplete, evals.NotFound, ValueError) as exc:
            raise ApplicationError(str(exc), non_retryable=True) from exc

    @activity.defn(name=rule.FAIL_ACTIVITY)
    async def fail(self, input: FailureInput) -> WorkflowResult:
        return await evals.fail_run(self._env.db, input.run_id, input.error)
