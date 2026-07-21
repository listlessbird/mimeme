from __future__ import annotations

import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session
from tests.factories import create_index_build, create_job, create_search_index_state

from mimeme.db.schema import Job, JobStatus, JobType, RebuildTrigger, SearchIndexState
from mimeme.domain.rebuild_claim import PrepareDecision, RebuildCoordinator

TIMEOUT = datetime.timedelta(minutes=180)


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _rebuild_job(session: Session, status: JobStatus = JobStatus.PENDING) -> str:
    job = create_job(session=session, type=JobType.REBUILD_INDEX, status=status)
    session.flush()
    return job.id


def _prepare(
    session: Session,
    *,
    trigger: RebuildTrigger,
    job_id: str | None = None,
    force: bool = False,
    now: datetime.datetime | None = None,
) -> PrepareDecision:
    return RebuildCoordinator(session).prepare(
        job_id=job_id,
        workflow_id="wf-scheduled",
        force=force,
        trigger=trigger,
        now=now or _now(),
        claim_timeout=TIMEOUT,
    )


def _job_status(session: Session, job_id: str) -> JobStatus:
    job = session.get(Job, job_id)
    assert job is not None
    return job.status


def test_scheduled_dirty_creates_job_and_claims(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=4, active_generation=1)

    decision = _prepare(db_session, trigger=RebuildTrigger.SCHEDULED)

    assert decision.decision == "build"
    assert decision.job_id is not None
    assert decision.target_generation == 4
    job = db_session.get(Job, decision.job_id)
    assert job is not None
    assert job.type == JobType.REBUILD_INDEX
    assert job.workflow_id == "wf-scheduled"
    state = db_session.get(SearchIndexState, 1)
    assert state is not None and state.rebuild_job_id == decision.job_id


def test_scheduled_clean_creates_no_job(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=3, active_generation=3)

    decision = _prepare(db_session, trigger=RebuildTrigger.SCHEDULED)

    assert decision.decision == "clean"
    assert decision.job_id is None
    assert db_session.scalars(select(Job)).first() is None


def test_scheduled_busy_creates_no_job(db_session: Session) -> None:
    owner = _rebuild_job(db_session, status=JobStatus.RUNNING)
    create_search_index_state(
        session=db_session,
        desired_generation=5,
        active_generation=1,
        rebuild_job_id=owner,
        rebuild_target_generation=5,
        rebuild_claimed_at=_now(),
    )

    decision = _prepare(db_session, trigger=RebuildTrigger.SCHEDULED)

    assert decision.decision == "busy"
    assert decision.job_id is None
    state = db_session.get(SearchIndexState, 1)
    assert state is not None and state.rebuild_job_id == owner


def test_manual_dirty_claims_existing_job(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=6, active_generation=2)
    job_id = _rebuild_job(db_session)

    decision = _prepare(db_session, trigger=RebuildTrigger.MANUAL, job_id=job_id)

    assert decision.decision == "build"
    assert decision.job_id == job_id
    assert decision.target_generation == 6
    state = db_session.get(SearchIndexState, 1)
    assert state is not None and state.rebuild_job_id == job_id


def test_manual_clean_non_force_completes_job_as_skipped(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=3, active_generation=3)
    create_index_build(session=db_session, version="v-cur", is_active=True, dimension=768)
    job_id = _rebuild_job(db_session)

    decision = _prepare(db_session, trigger=RebuildTrigger.MANUAL, job_id=job_id)

    assert decision.decision == "clean"
    job = db_session.get(Job, job_id)
    assert job is not None
    assert job.status == JobStatus.COMPLETED
    assert job.message == "Index already current"
    assert job.result is not None and '"skipped":true' in job.result


def test_manual_clean_without_active_build_completes_skipped(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=3, active_generation=3)
    job_id = _rebuild_job(db_session)

    decision = _prepare(db_session, trigger=RebuildTrigger.MANUAL, job_id=job_id)

    assert decision.decision == "clean"
    job = db_session.get(Job, job_id)
    assert job is not None
    assert job.status == JobStatus.COMPLETED
    assert job.result is not None and '"skipped":true' in job.result


def test_manual_clean_force_builds(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=3, active_generation=3)
    job_id = _rebuild_job(db_session)

    decision = _prepare(db_session, trigger=RebuildTrigger.MANUAL, job_id=job_id, force=True)

    assert decision.decision == "build"
    assert decision.target_generation == 3


def test_manual_busy_leaves_job_pending(db_session: Session) -> None:
    owner = _rebuild_job(db_session, status=JobStatus.RUNNING)
    create_search_index_state(
        session=db_session,
        desired_generation=5,
        active_generation=1,
        rebuild_job_id=owner,
        rebuild_target_generation=5,
        rebuild_claimed_at=_now(),
    )
    waiter = _rebuild_job(db_session)

    decision = _prepare(db_session, trigger=RebuildTrigger.MANUAL, job_id=waiter)

    assert decision.decision == "busy"
    assert decision.job_id == waiter
    assert _job_status(db_session, waiter) == JobStatus.PENDING


def test_manual_force_does_not_steal_live_claim(db_session: Session) -> None:
    owner = _rebuild_job(db_session, status=JobStatus.RUNNING)
    create_search_index_state(
        session=db_session,
        desired_generation=5,
        active_generation=1,
        rebuild_job_id=owner,
        rebuild_target_generation=5,
        rebuild_claimed_at=_now(),
    )
    waiter = _rebuild_job(db_session)

    decision = _prepare(db_session, trigger=RebuildTrigger.MANUAL, job_id=waiter, force=True)

    assert decision.decision == "busy"
    state = db_session.get(SearchIndexState, 1)
    assert state is not None and state.rebuild_job_id == owner
    assert _job_status(db_session, owner) == JobStatus.RUNNING


def test_non_expired_claim_is_not_stolen(db_session: Session) -> None:
    owner = _rebuild_job(db_session, status=JobStatus.RUNNING)
    create_search_index_state(
        session=db_session,
        desired_generation=5,
        active_generation=1,
        rebuild_job_id=owner,
        rebuild_target_generation=5,
        rebuild_claimed_at=_now() - datetime.timedelta(minutes=10),
    )

    decision = _prepare(db_session, trigger=RebuildTrigger.SCHEDULED)

    assert decision.decision == "busy"
    assert _job_status(db_session, owner) == JobStatus.RUNNING


def test_expired_claim_fails_owner_and_reclaims(db_session: Session) -> None:
    owner = _rebuild_job(db_session, status=JobStatus.RUNNING)
    create_search_index_state(
        session=db_session,
        desired_generation=5,
        active_generation=1,
        rebuild_job_id=owner,
        rebuild_target_generation=5,
        rebuild_claimed_at=_now() - datetime.timedelta(minutes=200),
    )

    decision = _prepare(db_session, trigger=RebuildTrigger.SCHEDULED)

    assert decision.decision == "build"
    assert decision.job_id is not None and decision.job_id != owner
    assert _job_status(db_session, owner) == JobStatus.FAILED
    state = db_session.get(SearchIndexState, 1)
    assert state is not None and state.rebuild_job_id == decision.job_id


def test_claim_expired_exactly_at_timeout_boundary_is_reclaimed(db_session: Session) -> None:
    owner = _rebuild_job(db_session, status=JobStatus.RUNNING)
    now = _now()
    create_search_index_state(
        session=db_session,
        desired_generation=5,
        active_generation=1,
        rebuild_job_id=owner,
        rebuild_target_generation=5,
        rebuild_claimed_at=now - TIMEOUT,
    )

    decision = _prepare(db_session, trigger=RebuildTrigger.SCHEDULED, now=now)

    assert decision.decision == "build"
    assert _job_status(db_session, owner) == JobStatus.FAILED


def test_expired_claim_manual_reclaims_for_manual_job(db_session: Session) -> None:
    owner = _rebuild_job(db_session, status=JobStatus.RUNNING)
    create_search_index_state(
        session=db_session,
        desired_generation=5,
        active_generation=1,
        rebuild_job_id=owner,
        rebuild_target_generation=5,
        rebuild_claimed_at=_now() - datetime.timedelta(minutes=200),
    )
    waiter = _rebuild_job(db_session)

    decision = _prepare(db_session, trigger=RebuildTrigger.MANUAL, job_id=waiter)

    assert decision.decision == "build"
    assert decision.job_id == waiter
    assert _job_status(db_session, owner) == JobStatus.FAILED
    state = db_session.get(SearchIndexState, 1)
    assert state is not None and state.rebuild_job_id == waiter


def test_terminal_owner_claim_is_recovered(db_session: Session) -> None:
    owner = _rebuild_job(db_session, status=JobStatus.COMPLETED)
    create_search_index_state(
        session=db_session,
        desired_generation=5,
        active_generation=1,
        rebuild_job_id=owner,
        rebuild_target_generation=5,
        rebuild_claimed_at=_now(),
    )

    decision = _prepare(db_session, trigger=RebuildTrigger.SCHEDULED)

    assert decision.decision == "build"
    assert decision.job_id != owner
