from __future__ import annotations

from sqlalchemy import func, select

from domain.image_ingest_input import ImageIngestInput, RemoteImageUrlInput
from domain.job_rules import (
    IndexBuildView,
    IngestJobCreation,
    JobCancellation,
    JobLifecycleNotFoundError,
    JobList,
    JobRowData,
    JobView,
    RebuildJobCreation,
    ensure_cancellable,
    mint_ingest_job,
    mint_rebuild_job,
    project_job,
)
from shared import db
from shared.models import IndexBuild, IngestURL, Job, JobStatus, JobType


class ApiJobStore:
    async def create_ingest_job(
        self,
        *,
        inputs: list[ImageIngestInput],
        dataset: str | None,
        tags: list[str],
        callback_url: str | None,
    ) -> IngestJobCreation:
        unique_inputs = list(dict.fromkeys(inputs))
        duplicates = len(inputs) - len(unique_inputs)
        job_id, workflow_id = mint_ingest_job()

        async with db.write_session() as session:
            job = Job(id=job_id, type=JobType.INGEST)
            session.add(job)
            await session.flush()

            for item in unique_inputs:
                if isinstance(item, RemoteImageUrlInput):
                    session.add(IngestURL(job_id=job_id, input_kind=item.kind, url=item.url))
                else:
                    session.add(
                        IngestURL(
                            job_id=job_id,
                            input_kind=item.kind,
                            artifact_key=item.artifact_key,
                        )
                    )

            await session.flush()

        return IngestJobCreation(
            job_id=job_id,
            workflow_id=workflow_id,
            queued=len(unique_inputs),
            duplicates=duplicates,
            dataset=dataset,
            tags=tags,
            callback_url=callback_url,
        )

    async def create_rebuild_job(
        self,
        *,
        force: bool,
        model_name: str,
        index_type: str,
    ) -> RebuildJobCreation:
        job_id, workflow_id = mint_rebuild_job()

        async with db.write_session() as session:
            job = Job(id=job_id, type=JobType.REBUILD_INDEX)
            session.add(job)
            await session.flush()
            view = project_job(JobRowData.model_validate(job))

        return RebuildJobCreation(
            job=view,
            workflow_id=workflow_id,
            force=force,
            model_name=model_name,
            index_type=index_type,
        )

    async def record_workflow_id(self, job_id: str, workflow_id: str) -> None:
        async with db.write_session() as session:
            job = await session.get(Job, job_id)
            if job is None:
                raise JobLifecycleNotFoundError(f"Job {job_id} not found")
            job.workflow_id = workflow_id
            await session.flush()

    async def get_job(self, job_id: str) -> JobView:
        async with db.read_session() as session:
            job = await session.get(Job, job_id)
            if job is None:
                raise JobLifecycleNotFoundError(f"Job {job_id} not found")
            return project_job(JobRowData.model_validate(job))

    async def list_jobs(
        self,
        *,
        status: JobStatus | None,
        job_type: JobType | None,
        limit: int,
    ) -> JobList:
        async with db.read_session() as session:
            stmt = select(Job)

            if status:
                stmt = stmt.where(Job.status == status)
            if job_type:
                stmt = stmt.where(Job.type == job_type)

            total = await session.scalar(select(func.count()).select_from(stmt.subquery())) or 0
            rows = await session.scalars(stmt.order_by(Job.created_at.desc()).limit(limit))

            jobs = rows.all()

            views = [project_job(JobRowData.model_validate(job)) for job in jobs]
            return JobList(jobs=views, total=total)

    async def list_index_builds(self, *, limit: int) -> list[IndexBuildView]:
        async with db.read_session() as session:
            rows = await session.scalars(
                select(IndexBuild).order_by(IndexBuild.created_at.desc()).limit(limit)
            )
            return [IndexBuildView.model_validate(row) for row in rows.all()]

    async def request_cancellation(self, job_id: str) -> JobCancellation:
        async with db.read_session() as session:
            job = await session.get(Job, job_id)
            if job is None:
                raise JobLifecycleNotFoundError(f"Job {job_id} not found")
            ensure_cancellable(job.status)
            return JobCancellation(workflow_id=job.workflow_id)

    async def mark_cancelled(self, job_id: str) -> None:
        async with db.write_session() as session:
            job = await session.get(Job, job_id)
            if job is None:
                raise JobLifecycleNotFoundError(f"Job {job_id} not found")
            job.status = JobStatus.CANCELLED
            await session.flush()
