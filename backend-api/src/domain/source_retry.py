from __future__ import annotations

import uuid
from collections.abc import Sequence

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from domain.source_item_browse import RunNotFoundError
from domain.source_registry import SourceNotFoundError
from shared import db
from shared.models import ProcessingStatus, SourceRunStatus
from shared.models.orm import IngestionSource, IngestURL, Job, JobType, SourceItem, SourceRun


class NothingToRetryError(Exception):
    pass


class SourceItemNotFoundError(Exception):
    pass


class RetryPlan(BaseModel, frozen=True):
    job_id: str
    workflow_id: str
    source_run_ids: list[int]
    dataset: str | None
    count: int


class SourceRetry:
    def retry_run(self, source_id: int, run_id: int) -> RetryPlan:

        with db.session_scope() as session:
            dataset = self._live_source_dataset_or_raise(session, source_id)
            run = session.execute(
                select(SourceRun).where(SourceRun.id == run_id, SourceRun.source_id == source_id)
            ).scalar_one_or_none()
            if run is None:
                raise RunNotFoundError(run_id)

            urls = (
                session.execute(
                    select(IngestURL).where(
                        IngestURL.source_run_id == run_id,
                        IngestURL.status == ProcessingStatus.FAILED,
                    )
                )
                .scalars()
                .all()
            )
            return self._reset_and_queue(session, urls, dataset)

    def retry_source(self, source_id: int) -> RetryPlan:

        with db.session_scope() as session:
            dataset = self._live_source_dataset_or_raise(session, source_id)
            urls = (
                session.execute(
                    select(IngestURL).where(
                        IngestURL.source_id == source_id,
                        IngestURL.status == ProcessingStatus.FAILED,
                    )
                )
                .scalars()
                .all()
            )
            return self._reset_and_queue(session, urls, dataset)

    def retry_item(self, source_id: int, source_item_id: int) -> RetryPlan:
        with db.session_scope() as session:
            dataset = self._live_source_dataset_or_raise(session, source_id)
            item = session.execute(
                select(SourceItem.id).where(
                    SourceItem.id == source_item_id, SourceItem.source_id == source_id
                )
            ).scalar_one_or_none()
            if item is None:
                raise SourceItemNotFoundError(source_item_id)

            urls = (
                session.execute(
                    select(IngestURL).where(
                        IngestURL.source_item_id == source_item_id,
                        IngestURL.status == ProcessingStatus.FAILED,
                    )
                )
                .scalars()
                .all()
            )
            return self._reset_and_queue(session, urls, dataset)

    def _live_source_dataset_or_raise(self, session: Session, source_id: int) -> str | None:
        source = session.execute(
            select(IngestionSource).where(
                IngestionSource.id == source_id, IngestionSource.deleted_at.is_(None)
            )
        ).scalar_one_or_none()
        if source is None:
            raise SourceNotFoundError(source_id)
        return source.dataset

    def _reset_and_queue(
        self, session: Session, urls: Sequence[IngestURL], dataset: str | None
    ) -> RetryPlan:
        if not urls:
            raise NothingToRetryError

        job_id = f"ingest-{uuid.uuid4().hex[:12]}"
        session.add(Job(id=job_id, type=JobType.INGEST))
        session.flush()

        run_ids: list[int] = []
        for url in urls:
            url.job_id = job_id
            url.status = ProcessingStatus.PENDING
            url.error_message = None
            url.image_id = None
            url.duplicate_reason = None
            url.duplicate_of_image_id = None
            if url.source_run_id is not None and url.source_run_id not in run_ids:
                run_ids.append(url.source_run_id)

        for run_id in run_ids:
            run = session.get(SourceRun, run_id)
            if run is not None:
                run.status = SourceRunStatus.RUNNING

        return RetryPlan(
            job_id=job_id,
            workflow_id=f"source-retry-workflow-{job_id}",
            source_run_ids=run_ids,
            dataset=dataset,
            count=len(urls),
        )
