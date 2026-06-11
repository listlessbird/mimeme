"""Tests for finalize_source_run_activity (issue 03, slice 4).

Finalization sets the run's *stored* status + completed_at, derived by querying
its records (source_items + ingest_urls) via derive_run_accounting. Counts stay
derived-on-read; only status/completed_at are persisted. It must be idempotent so
an activity retry after ingest finished does not corrupt the recorded outcome.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy.orm import Session
from temporalio.testing import ActivityEnvironment

from activities.ingestion.activities import finalize_source_run_activity
from activities.ingestion.models import FinalizeSourceRunInput
from shared.models import DuplicateReason, ProcessingStatus, SourceRunStatus
from tests.factories import (
    create_ingest_url,
    create_ingestion_source,
    create_job,
    create_source_item,
    create_source_run,
)


@pytest.fixture()
def activity_env() -> ActivityEnvironment:
    return ActivityEnvironment()


def _run(session: Session) -> Any:
    source = create_ingestion_source(session=session, adapter_key="meme_api")
    return create_source_run(session=session, source=source, source_id=source.id)


def _url(session: Session, run: Any, **kwargs: object) -> None:
    job = create_job(session=session)
    create_ingest_url(
        session=session,
        job=job,
        source_id=run.source_id,
        source_run_id=run.id,
        **kwargs,
    )


def _finalize(activity_env: ActivityEnvironment, run: Any) -> Any:
    return activity_env.run(
        finalize_source_run_activity, FinalizeSourceRunInput(source_run_id=run.id)
    )


@pytest.mark.usefixtures("_patch_session_scope")
class TestFinalizeSourceRunActivity:
    def test_all_done_is_completed_with_derived_counts(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        run = _run(db_session)
        create_source_item(session=db_session, source_id=run.source_id, last_source_run_id=run.id)
        create_source_item(session=db_session, source_id=run.source_id, last_source_run_id=run.id)
        _url(db_session, run, status=ProcessingStatus.DONE)
        _url(
            db_session,
            run,
            status=ProcessingStatus.DONE,
            duplicate_reason=DuplicateReason.SHA256,
        )

        result = _finalize(activity_env, run)

        assert result.status == SourceRunStatus.COMPLETED
        assert result.discovered == 2
        assert result.queued == 2
        assert result.duplicate == 1
        assert result.failed == 0

        db_session.refresh(run)
        assert run.status == SourceRunStatus.COMPLETED
        assert run.completed_at is not None

    def test_some_failed_is_partial(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        run = _run(db_session)
        _url(db_session, run, status=ProcessingStatus.DONE)
        _url(db_session, run, status=ProcessingStatus.FAILED)

        result = _finalize(activity_env, run)

        assert result.status == SourceRunStatus.PARTIAL
        assert result.failed == 1

    def test_all_failed_is_failed(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        run = _run(db_session)
        _url(db_session, run, status=ProcessingStatus.FAILED)
        _url(db_session, run, status=ProcessingStatus.FAILED)

        result = _finalize(activity_env, run)

        assert result.status == SourceRunStatus.FAILED

    def test_no_queued_urls_is_completed(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        run = _run(db_session)

        result = _finalize(activity_env, run)

        assert result.status == SourceRunStatus.COMPLETED
        assert result.queued == 0

    def test_retry_is_idempotent(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        run = _run(db_session)
        _url(db_session, run, status=ProcessingStatus.DONE)

        first = _finalize(activity_env, run)
        db_session.refresh(run)
        completed_at = run.completed_at

        second = _finalize(activity_env, run)
        db_session.refresh(run)

        assert second.status == first.status == SourceRunStatus.COMPLETED
        # completed_at is set once, not bumped on the retry.
        assert run.completed_at == completed_at
