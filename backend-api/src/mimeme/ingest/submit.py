from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from temporalio.client import Client

from mimeme import storage
from mimeme.db import Db
from mimeme.ingest import rule
from mimeme.ingest.model import ItemRef, Staged, Submission, WorkflowInput
from mimeme.ingest.workflow import IngestWorkflow
from mimeme.job import ops as job_ops
from mimeme.job.model import IngestCreation
from mimeme.job.store import Store as JobStore


class Deps(Protocol):
    db: Db
    temporal: Client
    artifacts: storage.Store


async def submit(env: Deps, submission: Submission) -> IngestCreation:
    creation = await job_ops.create_ingest(
        env.db,
        inputs=list(submission.urls),
        dataset=submission.dataset,
        tags=submission.tags,
        callback_url=submission.callback_url,
    )
    wf_id = rule.workflow_id(creation.job_id)
    async with env.db.read_session() as session:
        init = await JobStore(session).ingest_urls(creation.job_id)
    items = [ItemRef(item_id=ref.id, source=ref.input) for ref in init.urls]
    await env.temporal.start_workflow(
        IngestWorkflow.run,
        WorkflowInput(job_id=creation.job_id, dataset=creation.dataset, items=items),
        id=wf_id,
        task_queue=rule.TASK_QUEUE,
    )
    await job_ops.record_workflow_id(env.db, creation.job_id, wf_id)
    await job_ops.start(env.db, creation.job_id)
    return creation.model_copy(update={"workflow_id": wf_id})


async def stage_upload(
    env: Deps, *, content: bytes, filename: str | None, content_type: str | None
) -> Staged:
    key = rule.upload_staging_key(filename)
    await env.artifacts.put(
        storage.Object(key),
        _once(content),
        length=len(content),
        content_type=content_type or "application/octet-stream",
        checksum=storage.Checksum.of(content),
    )
    return Staged(artifact_key=key)


async def _once(data: bytes) -> AsyncIterator[bytes]:
    yield data
