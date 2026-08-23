from __future__ import annotations

import uuid
from typing import Protocol

from temporalio.client import Client

from mimeme.config import Settings
from mimeme.db import Db
from mimeme.search_eval import rule
from mimeme.search_eval import service as evals
from mimeme.search_eval.model import RunMode, RunView, WorkflowInput
from mimeme.search_eval.workflow import SearchEvalScoreWorkflow, SearchEvalWorkflow


class Deps(Protocol):
    db: Db
    temporal: Client
    settings: Settings


async def submit_run(env: Deps, *, mode: RunMode) -> RunView:
    run_id = uuid.uuid4().hex
    workflow_id = rule.run_workflow_id(run_id)
    run = await evals.create_run(
        env.db,
        run_id=run_id,
        mode=mode,
        workflow_id=workflow_id,
    )
    try:
        await env.temporal.start_workflow(
            SearchEvalWorkflow.run,
            WorkflowInput(run_id=run.id),
            id=workflow_id,
            task_queue=env.settings.temporal.task_queue,
        )
    except Exception as failure:
        await evals.fail_run(env.db, run.id, str(failure))
        raise
    return run


async def submit_rescore(env: Deps, run_id: str) -> RunView:
    workflow_id = rule.score_workflow_id(run_id, uuid.uuid4().hex[:12])
    run = await evals.queue_rescore(env.db, run_id, workflow_id=workflow_id)
    try:
        await env.temporal.start_workflow(
            SearchEvalScoreWorkflow.run,
            WorkflowInput(run_id=run.id),
            id=workflow_id,
            task_queue=env.settings.temporal.task_queue,
        )
    except Exception as failure:
        await evals.fail_run(env.db, run.id, str(failure))
        raise
    return run
