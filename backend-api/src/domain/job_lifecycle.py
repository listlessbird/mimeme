from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from domain.image_ingest_input import ImageIngestInput, restore_image_ingest_input
from domain.job_rules import (
    INGEST_URL_ERROR_LIMIT,
    JOB_ERROR_LIMIT,
    IngestJobCreation,
    IngestJobResultPayload,
    JobLifecycleNotFoundError,
    JobRowData,
    RebuildJobCreation,
    RebuildJobResultPayload,
    derive_completion_status,
    mint_ingest_job,
    mint_rebuild_job,
    project_job,
    truncate_error,
)
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


class SourceIngestItem(BaseModel, frozen=True):
    url: str
    source_id: int
    source_run_id: int
    source_item_id: int


class IngestUrlRef(BaseModel, frozen=True):
    id: int
    input: ImageIngestInput


class IngestInitialization(BaseModel, frozen=True):
    urls: list[IngestUrlRef]


class IngestUrlDoneResult(BaseModel, frozen=True):
    found: bool
    image_exists: bool | None


class JobLifecycle:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create_source_ingest_job(
        self,
        *,
        dataset: str | None,
        items: list[SourceIngestItem],
    ) -> IngestJobCreation:
        job_id, workflow_id = mint_ingest_job()

        job = Job(id=job_id, type=JobType.INGEST)
        self._db.add(job)
        self._db.flush()

        for item in items:
            self._db.add(
                IngestURL(
                    job_id=job_id,
                    input_kind="remote_image_url",
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
        job_id, workflow_id = mint_rebuild_job()
        job = Job(id=job_id, type=JobType.REBUILD_INDEX)
        self._db.add(job)
        self._db.commit()

        return RebuildJobCreation(
            job=project_job(JobRowData.model_validate(job)),
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

    def initialize_ingest(self, job_id: str) -> IngestInitialization:
        self.start_job(job_id)
        urls = self._db.scalars(select(IngestURL).where(IngestURL.job_id == job_id)).all()
        return IngestInitialization(
            urls=[
                IngestUrlRef(
                    id=row.id,
                    input=restore_image_ingest_input(
                        kind=row.input_kind,
                        url=row.url,
                        artifact_key=row.artifact_key,
                    ),
                )
                for row in urls
            ]
        )

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
        job.status = derive_completion_status(failed=failed)
        job.progress = 100.0
        job.completed_at = datetime.now(UTC)
        job.result = IngestJobResultPayload(
            processed=processed, failed=failed, duplicates=duplicates
        ).model_dump_json()
        self._db.flush()
        return True

    def fail_rebuild_job(self, job_id: str, error: str) -> bool:
        job = self._db.get(Job, job_id)
        if job is None:
            return False
        job.status = JobStatus.FAILED
        job.message = truncate_error(error, JOB_ERROR_LIMIT)
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
        job.result = RebuildJobResultPayload(
            version=version,
            num_vectors=num_vectors,
            dimension=dimension,
            removed_versions=removed_versions,
            text_num_vectors=text_num_vectors,
        ).model_dump_json()

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
        url.error_message = truncate_error(error, INGEST_URL_ERROR_LIMIT)
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
