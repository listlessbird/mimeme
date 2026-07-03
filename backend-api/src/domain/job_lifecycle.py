from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from shared.models import (
    DuplicateReason,
    IngestStage,
    IngestURL,
    Job,
    JobStatus,
    JobType,
    ProcessingStatus,
)
from shared.models import ORMImage as Image


class JobLifecycleNotFoundError(Exception):
    pass


class JobLifecycleInvalidStateError(Exception):
    pass


class JobView(BaseModel, frozen=True):
    id: str
    type: JobType
    status: JobStatus
    progress: float
    message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    result: dict[str, Any] | None


class JobList(BaseModel, frozen=True):
    jobs: list[JobView]
    total: int


class IngestJobCreation(BaseModel, frozen=True):
    job_id: str
    workflow_id: str
    queued: int
    duplicates: int
    dataset: str | None
    tags: list[str]
    callback_url: str | None


class RebuildJobCreation(BaseModel, frozen=True):
    job: JobView
    workflow_id: str
    force: bool
    model_name: str
    index_type: str


class SourceIngestItem(BaseModel, frozen=True):
    url: str
    source_id: int
    source_run_id: int
    source_item_id: int


class IngestUrlRef(BaseModel, frozen=True):
    id: int
    url: str


class IngestInitialization(BaseModel, frozen=True):
    urls: list[IngestUrlRef]


class IngestUrlDoneResult(BaseModel, frozen=True):
    found: bool
    image_exists: bool | None


class JobCancellation(BaseModel, frozen=True):
    workflow_id: str | None


class JobLifecycle:
    def __init__(self, db: Session) -> None:
        self._db = db

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

        job = Job(id=job_id, type=JobType.INGEST)
        self._db.add(job)
        self._db.flush()

        for url in unique_urls:
            self._db.add(IngestURL(job_id=job_id, url=url))

        self._db.commit()
        return IngestJobCreation(
            job_id=job_id,
            workflow_id=workflow_id,
            queued=len(unique_urls),
            duplicates=duplicates,
            dataset=dataset,
            tags=tags,
            callback_url=callback_url,
        )

    def create_source_ingest_job(
        self,
        *,
        dataset: str | None,
        items: list[SourceIngestItem],
    ) -> IngestJobCreation:
        """Create the INGEST Job + provenance-carrying ingest_urls for a Source run.

        Unlike ``create_ingest_job`` (the manual path), this only ``flush``es —
        the calling activity owns the transaction boundary via ``session_scope``.
        """
        job_id = f"ingest-{uuid.uuid4().hex[:12]}"
        workflow_id = f"ingest-workflow-{job_id}"

        job = Job(id=job_id, type=JobType.INGEST)
        self._db.add(job)
        self._db.flush()

        for item in items:
            self._db.add(
                IngestURL(
                    job_id=job_id,
                    url=item.url,
                    source_id=item.source_id,
                    source_run_id=item.source_run_id,
                    source_item_id=item.source_item_id,
                )
            )

        self._db.flush()
        return IngestJobCreation(
            job_id=job_id,
            workflow_id=workflow_id,
            queued=len(items),
            duplicates=0,
            dataset=dataset,
            tags=[],
            callback_url=None,
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
        job = Job(id=job_id, type=JobType.REBUILD_INDEX)
        self._db.add(job)
        self._db.commit()

        return RebuildJobCreation(
            job=self._project_job(job),
            workflow_id=workflow_id,
            force=force,
            model_name=model_name,
            index_type=index_type,
        )

    def record_workflow_id(self, job_id: str, workflow_id: str) -> None:
        job = self._db.get(Job, job_id)
        if job is None:
            raise JobLifecycleNotFoundError(f"Job {job_id} not found")
        job.workflow_id = workflow_id
        self._db.commit()

    def get_job(self, job_id: str) -> JobView:
        job = self._db.get(Job, job_id)
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
        stmt = select(Job)

        if status:
            stmt = stmt.where(Job.status == status)
        if job_type:
            stmt = stmt.where(Job.type == job_type)

        total = self._db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        jobs = self._db.scalars(stmt.order_by(Job.created_at.desc()).limit(limit)).all()
        return JobList(jobs=[self._project_job(job) for job in jobs], total=total)

    def request_cancellation(self, job_id: str) -> JobCancellation:
        job = self._db.get(Job, job_id)
        if job is None:
            raise JobLifecycleNotFoundError(f"Job {job_id} not found")
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            raise JobLifecycleInvalidStateError("Cannot cancel completed job")
        return JobCancellation(workflow_id=job.workflow_id)

    def mark_cancelled(self, job_id: str) -> None:
        job = self._db.get(Job, job_id)
        if job is None:
            raise JobLifecycleNotFoundError(f"Job {job_id} not found")
        job.status = JobStatus.CANCELLED
        self._db.commit()

    def initialize_ingest(self, job_id: str) -> IngestInitialization:
        self.start_job(job_id)
        urls = self._db.scalars(select(IngestURL).where(IngestURL.job_id == job_id)).all()
        return IngestInitialization(urls=[IngestUrlRef(id=url.id, url=url.url) for url in urls])

    def start_job(self, job_id: str) -> None:
        job = self._db.get(Job, job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(UTC)
        self._db.flush()

    def update_progress(self, job_id: str, progress: float, message: str | None = None) -> bool:
        job = self._db.get(Job, job_id)
        if job is None:
            return False
        job.progress = progress
        if message is not None:
            job.message = message
        self._db.flush()
        return True

    def complete_ingest_job(
        self,
        *,
        job_id: str,
        processed: int,
        failed: int,
        duplicates: int,
    ) -> bool:
        job = self._db.get(Job, job_id)
        if job is None:
            return False
        job.status = JobStatus.COMPLETED if failed == 0 else JobStatus.FAILED
        job.progress = 100.0
        job.completed_at = datetime.now(UTC)
        job.result = json.dumps(
            {
                "processed": processed,
                "failed": failed,
                "duplicates": duplicates,
            }
        )
        self._db.flush()
        return True

    def fail_rebuild_job(self, job_id: str, error: str) -> bool:
        job = self._db.get(Job, job_id)
        if job is None:
            return False
        job.status = JobStatus.FAILED
        job.message = error[:2000] if error else error
        job.completed_at = datetime.now(UTC)
        self._db.flush()
        return True

    def complete_rebuild_job(
        self,
        *,
        job_id: str,
        version: str,
        num_vectors: int,
        dimension: int,
        removed_versions: list[str],
        text_num_vectors: int | None,
    ) -> None:
        job = self._db.get(Job, job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        job.status = JobStatus.COMPLETED
        job.progress = 100.0
        job.completed_at = datetime.now(UTC)
        job.result = json.dumps(
            {
                "version": version,
                "num_vectors": num_vectors,
                "dimension": dimension,
                "removed_versions": removed_versions,
                "text_num_vectors": text_num_vectors,
            }
        )
        self._db.flush()

    def record_stage(self, ingest_url_id: int, stage: IngestStage) -> bool:
        url = self._db.get(IngestURL, ingest_url_id)
        if url is None:
            return False
        url.stage = stage
        url.stage_updated_at = datetime.now(UTC)
        self._db.flush()
        return True

    def mark_ingest_url_failed(self, ingest_url_id: int, error: str) -> bool:
        url = self._db.get(IngestURL, ingest_url_id)
        if url is None:
            return False
        url.status = ProcessingStatus.FAILED
        url.error_message = error[:1000] if error else error
        self._db.flush()
        return True

    def mark_ingest_url_done(
        self,
        ingest_url_id: int,
        image_id: int,
        duplicate_reason: DuplicateReason | None = None,
        duplicate_of_image_id: int | None = None,
    ) -> IngestUrlDoneResult:
        url = self._db.get(IngestURL, ingest_url_id)
        if url is None:
            return IngestUrlDoneResult(found=False, image_exists=None)

        image_exists = self._db.scalar(select(Image.id).where(Image.id == image_id)) is not None

        if image_exists:
            url.status = ProcessingStatus.DONE
            url.image_id = image_id
            url.duplicate_reason = duplicate_reason
            url.duplicate_of_image_id = duplicate_of_image_id
        else:
            url.status = ProcessingStatus.FAILED
            url.error_message = f"Image {image_id} not found while marking ingest URL done"
        self._db.flush()
        return IngestUrlDoneResult(found=True, image_exists=image_exists)

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
            result=self._parse_result(job.result),
        )

    def _parse_result(self, result: str | None) -> dict[str, Any] | None:
        if result is None:
            return None
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return {"raw": result}
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
