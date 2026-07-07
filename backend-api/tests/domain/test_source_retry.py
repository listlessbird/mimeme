"""Tests for SourceRetry — re-enqueueing failed Source ingest attempts.

Retry resets the *existing* failed ``IngestURL`` rows in place (preserving the
one-attempt-per-source-item invariant the galleries depend on), reattaches them
to a fresh INGEST ``Job`` for the pipeline to pick up, and marks the affected
runs RUNNING. Run accounting is re-derived later by finalize, not here.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session
from tests.factories import (
    create_image,
    create_ingest_url,
    create_ingestion_source,
    create_job,
    create_source_item,
    create_source_run,
)

from domain.source_item_browse import RunNotFoundError
from domain.source_registry import SourceNotFoundError
from domain.source_retry import (
    NothingToRetryError,
    SourceItemNotFoundError,
    SourceRetry,
)
from shared.models import ProcessingStatus, SourceRunStatus

pytestmark = pytest.mark.usefixtures("_patch_domain_session_scope")


def _failed_url(session: Session, run, item, **kwargs):
    job = create_job(session=session)
    return create_ingest_url(
        session=session,
        job=job,
        source_id=run.source_id,
        source_run_id=run.id,
        source_item_id=item.id,
        status=ProcessingStatus.FAILED,
        error_message="boom",
        **kwargs,
    )


class TestRetryRun:
    def test_resets_failed_urls_and_queues_fresh_job(self, db_session: Session) -> None:
        src = create_ingestion_source(session=db_session, dataset="memes")
        run = create_source_run(session=db_session, source=src, source_id=src.id)
        item = create_source_item(session=db_session, source=src, source_id=src.id)
        url = _failed_url(db_session, run, item)

        plan = SourceRetry().retry_run(src.id, run.id)

        db_session.refresh(url)
        assert url.status == ProcessingStatus.PENDING
        assert url.error_message is None
        assert url.job_id == plan.job_id
        assert plan.count == 1
        assert plan.source_run_ids == [run.id]
        assert plan.dataset == "memes"
        assert plan.workflow_id == f"source-retry-workflow-{plan.job_id}"
        assert plan.workflow_id != f"ingest-workflow-{plan.job_id}"

    def test_leaves_succeeded_urls_untouched(self, db_session: Session) -> None:
        src = create_ingestion_source(session=db_session)
        run = create_source_run(session=db_session, source=src, source_id=src.id)
        item = create_source_item(session=db_session, source=src, source_id=src.id)
        other = create_source_item(session=db_session, source=src, source_id=src.id)
        image = create_image(session=db_session)
        done_job = create_job(session=db_session)
        done = create_ingest_url(
            session=db_session,
            job=done_job,
            source_id=run.source_id,
            source_run_id=run.id,
            source_item_id=other.id,
            status=ProcessingStatus.DONE,
            image_id=image.id,
        )
        failed = _failed_url(db_session, run, item)

        plan = SourceRetry().retry_run(src.id, run.id)

        db_session.refresh(done)
        db_session.refresh(failed)
        assert done.status == ProcessingStatus.DONE
        assert done.job_id == done_job.id
        assert failed.job_id == plan.job_id
        assert plan.count == 1

    def test_marks_affected_run_running(self, db_session: Session) -> None:
        src = create_ingestion_source(session=db_session)
        run = create_source_run(
            session=db_session, source=src, source_id=src.id, status=SourceRunStatus.PARTIAL
        )
        item = create_source_item(session=db_session, source=src, source_id=src.id)
        _failed_url(db_session, run, item)

        SourceRetry().retry_run(src.id, run.id)

        db_session.refresh(run)
        assert run.status == SourceRunStatus.RUNNING

    def test_no_failed_items_raises(self, db_session: Session) -> None:
        src = create_ingestion_source(session=db_session)
        run = create_source_run(session=db_session, source=src, source_id=src.id)

        with pytest.raises(NothingToRetryError):
            SourceRetry().retry_run(src.id, run.id)

    def test_unknown_run_raises(self, db_session: Session) -> None:
        src = create_ingestion_source(session=db_session)

        with pytest.raises(RunNotFoundError):
            SourceRetry().retry_run(src.id, 999999)

    def test_unknown_source_raises(self, db_session: Session) -> None:
        with pytest.raises(SourceNotFoundError):
            SourceRetry().retry_run(999999, 1)


class TestRetrySource:
    def test_resets_failed_across_all_runs(self, db_session: Session) -> None:
        src = create_ingestion_source(session=db_session, dataset="memes")
        run_a = create_source_run(session=db_session, source=src, source_id=src.id)
        run_b = create_source_run(session=db_session, source=src, source_id=src.id)
        item_a = create_source_item(session=db_session, source=src, source_id=src.id)
        item_b = create_source_item(session=db_session, source=src, source_id=src.id)
        url_a = _failed_url(db_session, run_a, item_a)
        url_b = _failed_url(db_session, run_b, item_b)

        plan = SourceRetry().retry_source(src.id)

        db_session.refresh(url_a)
        db_session.refresh(url_b)
        assert url_a.status == ProcessingStatus.PENDING
        assert url_b.status == ProcessingStatus.PENDING
        assert url_a.job_id == plan.job_id
        assert plan.count == 2
        assert sorted(plan.source_run_ids) == sorted([run_a.id, run_b.id])

    def test_no_failures_raises(self, db_session: Session) -> None:
        src = create_ingestion_source(session=db_session)

        with pytest.raises(NothingToRetryError):
            SourceRetry().retry_source(src.id)

    def test_unknown_source_raises(self, db_session: Session) -> None:
        with pytest.raises(SourceNotFoundError):
            SourceRetry().retry_source(999999)


class TestRetryItem:
    def test_resets_single_items_failed_attempt(self, db_session: Session) -> None:
        src = create_ingestion_source(session=db_session, dataset="memes")
        run = create_source_run(session=db_session, source=src, source_id=src.id)
        item = create_source_item(session=db_session, source=src, source_id=src.id)
        url = _failed_url(db_session, run, item)

        plan = SourceRetry().retry_item(src.id, item.id)

        db_session.refresh(url)
        assert url.status == ProcessingStatus.PENDING
        assert url.job_id == plan.job_id
        assert plan.count == 1
        assert plan.source_run_ids == [run.id]
        assert plan.dataset == "memes"

    def test_unknown_item_raises(self, db_session: Session) -> None:
        src = create_ingestion_source(session=db_session)

        with pytest.raises(SourceItemNotFoundError):
            SourceRetry().retry_item(src.id, 999999)

    def test_item_with_no_failed_attempt_raises(self, db_session: Session) -> None:
        src = create_ingestion_source(session=db_session)
        run = create_source_run(session=db_session, source=src, source_id=src.id)
        item = create_source_item(session=db_session, source=src, source_id=src.id)
        image = create_image(session=db_session)
        job = create_job(session=db_session)
        create_ingest_url(
            session=db_session,
            job=job,
            source_id=run.source_id,
            source_run_id=run.id,
            source_item_id=item.id,
            status=ProcessingStatus.DONE,
            image_id=image.id,
        )

        with pytest.raises(NothingToRetryError):
            SourceRetry().retry_item(src.id, item.id)
