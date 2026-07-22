from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock

from sqlalchemy import select

from mimeme import storage
from mimeme.db.schema import IngestURL, Job, JobStatus, JobType
from mimeme.ingest import rule
from mimeme.ingest.model import RemoteUrl, Staged, Submission
from mimeme.ingest.submit import stage_upload, submit
from mimeme.ingest.workflow import IngestWorkflow
from tests.job.conftest import SavepointDb
from tests.support.storage import Memory


@dataclass
class SubmitEnv:
    db: SavepointDb
    temporal: AsyncMock
    artifacts: Memory


def _env(db: SavepointDb) -> SubmitEnv:
    return SubmitEnv(db=db, temporal=AsyncMock(), artifacts=Memory())


class TestSubmit:
    async def test_url_submission_creates_job_and_starts_v2_workflow(self, db: SavepointDb) -> None:
        env = _env(db)
        creation = await submit(
            env,
            Submission(
                urls=[RemoteUrl(url="https://a/1.jpg"), RemoteUrl(url="https://a/2.png")],
                dataset="memes",
            ),
        )
        assert creation.queued == 2
        assert creation.workflow_id == rule.workflow_id(creation.job_id)

        env.temporal.start_workflow.assert_awaited_once()
        call = env.temporal.start_workflow.await_args
        assert call.args[0] == IngestWorkflow.run
        wf_input = call.args[1]
        assert wf_input.job_id == creation.job_id
        assert {ref.item_id for ref in wf_input.items} and len(wf_input.items) == 2
        assert call.kwargs["id"] == rule.workflow_id(creation.job_id)
        assert call.kwargs["task_queue"] == rule.TASK_QUEUE

        async with db.read_session() as session:
            job = await session.get(Job, creation.job_id)
            assert job.type is JobType.INGEST and job.status is JobStatus.RUNNING
            assert job.workflow_id == rule.workflow_id(creation.job_id)
            urls = (
                await session.scalars(select(IngestURL).where(IngestURL.job_id == creation.job_id))
            ).all()
            assert len(urls) == 2

    async def test_staged_submission_resolves_source(self, db: SavepointDb) -> None:
        env = _env(db)
        creation = await submit(
            env, Submission(urls=[Staged(artifact_key="uploads/staging/x.png")])
        )
        call = env.temporal.start_workflow.await_args
        [ref] = call.args[1].items
        assert isinstance(ref.source, Staged)
        assert ref.source.artifact_key == "uploads/staging/x.png"
        assert creation.queued == 1


class TestStageUpload:
    async def test_stages_bytes_to_artifacts(self, db: SavepointDb) -> None:
        env = _env(db)
        staged = await stage_upload(
            env, content=b"imagedata", filename="cat.PNG", content_type="image/png"
        )
        assert isinstance(staged, Staged)
        assert staged.artifact_key.startswith(rule.UPLOAD_STAGING_PREFIX)
        assert staged.artifact_key.endswith(".png")
        assert await env.artifacts.stat(storage.Object(staged.artifact_key)) is not None
