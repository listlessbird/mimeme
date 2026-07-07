from __future__ import annotations

import uuid

from pydantic import ValidationError
from sqlalchemy import func, select

from domain.job_lifecycle import (
    IngestJobCreation,
    JobCancellation,
    JobLifecycleInvalidStateError,
    JobLifecycleNotFoundError,
    JobList,
    JobView,
    RebuildJobCreation,
)
from domain.job_rules import (
    IngestJobResultPayload,
    JobResultPayload,
    RawJobResultPayload,
    RebuildJobResultPayload,
)
from shared import db
from shared.models import IngestURL, Job, JobStatus, JobType


class ApiJobStore:
    def create_ingest_job(
        self,
        *,
        urls: list[str],
        dataset: str | None,
        tags: list[str],
        callback_url: str | None,
    ) -> IngestJobCreation:
        unique_urls = list(dict.fromkeys(urls))
        duplicates = len(urls) - len(unique_urls)
        job_id = f"ingest-{uuid.uuid4().hex[:12]}"
        workflow_id = f"ingest-workflow-{job_id}"

        with db.session_scope() as session:
            job = Job(id=job_id, type=JobType.INGEST)
            session.add(job)
            session.flush()

            for url in unique_urls:
                session.add(IngestURL(job_id=job_id, url=url))

            session.flush()

        return IngestJobCreation(
            job_id=job_id,
            workflow_id=workflow_id,
            queued=len(unique_urls),
            duplicates=duplicates,
            dataset=dataset,
            tags=tags,
            callback_url=callback_url,
        )

    def create_rebuild_job(
        self,
        *,
        force: bool,
        model_name: str,
        index_type: str,
    ) -> RebuildJobCreation:
        job_id = f"rebuild-{uuid.uuid4().hex[:12]}"
        workflow_id = f"rebuild-workflow-{job_id}"

        with db.session_scope() as session:
            job = Job(id=job_id, type=JobType.REBUILD_INDEX)
            session.add(job)
            session.flush()
            view = self._project_job(job)

        return RebuildJobCreation(
            job=view,
            workflow_id=workflow_id,
            force=force,
            model_name=model_name,
            index_type=index_type,
        )

    def record_workflow_id(self, job_id: str, workflow_id: str) -> None:
        with db.session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise JobLifecycleNotFoundError(f"Job {job_id} not found")
            job.workflow_id = workflow_id
            session.flush()

    def get_job(self, job_id: str) -> JobView:
        with db.read_session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise JobLifecycleNotFoundError(f"Job {job_id} not found")
            return self._project_job(job)

    def list_jobs(
        self,
        *,
        status: JobStatus | None,
        job_type: JobType | None,
        limit: int,
    ) -> JobList:
        with db.read_session_scope() as session:
            stmt = select(Job)

            if status:
                stmt = stmt.where(Job.status == status)
            if job_type:
                stmt = stmt.where(Job.type == job_type)

            total = session.scalar(select(func.count()).select_from(stmt.subquery())) or 0
            jobs = session.scalars(stmt.order_by(Job.created_at.desc()).limit(limit)).all()
            return JobList(jobs=[self._project_job(job) for job in jobs], total=total)

    def request_cancellation(self, job_id: str) -> JobCancellation:
        with db.read_session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise JobLifecycleNotFoundError(f"Job {job_id} not found")
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                raise JobLifecycleInvalidStateError("Cannot cancel completed job")
            return JobCancellation(workflow_id=job.workflow_id)

    def mark_cancelled(self, job_id: str) -> None:
        with db.session_scope() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise JobLifecycleNotFoundError(f"Job {job_id} not found")
            job.status = JobStatus.CANCELLED
            session.flush()

    def _project_job(self, job: Job) -> JobView:
        return JobView(
            id=job.id,
            type=job.type,
            status=job.status,
            progress=job.progress,
            message=job.message,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            result=self._parse_result(job.type, job.result),
        )

    def _parse_result(self, job_type: JobType, result: str | None) -> JobResultPayload | None:
        if result is None:
            return None

        try:
            if job_type == JobType.REBUILD_INDEX:
                return RebuildJobResultPayload.model_validate_json(result)
            return IngestJobResultPayload.model_validate_json(result)
        except ValidationError:
            return RawJobResultPayload(raw=result)
