from __future__ import annotations

import datetime

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from shared.models import Job, JobType, SearchIndexState
from tests.factories import create_job, create_search_index_state


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def test_singleton_row_defaults(db_session: Session) -> None:
    state = create_search_index_state(session=db_session)
    db_session.refresh(state)

    assert state.id == 1
    assert state.desired_generation == 1
    assert state.active_generation == 0
    assert state.rebuild_job_id is None
    assert state.rebuild_target_generation is None
    assert state.rebuild_claimed_at is None
    assert state.created_at is not None
    assert state.updated_at is not None


def test_id_must_be_one(db_session: Session) -> None:
    db_session.add(SearchIndexState(id=2, desired_generation=1, active_generation=0))
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_negative_generation_rejected(db_session: Session) -> None:
    db_session.add(SearchIndexState(id=1, desired_generation=-1, active_generation=0))
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_active_greater_than_desired_rejected(db_session: Session) -> None:
    db_session.add(SearchIndexState(id=1, desired_generation=1, active_generation=2))
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


@pytest.mark.parametrize(
    "present", ["rebuild_job_id", "rebuild_target_generation", "rebuild_claimed_at"]
)
def test_partial_claim_rejected(db_session: Session, present: str) -> None:
    job = create_job(session=db_session, type=JobType.REBUILD_INDEX)
    db_session.flush()
    values = {
        "rebuild_job_id": job.id,
        "rebuild_target_generation": 1,
        "rebuild_claimed_at": _now(),
    }
    db_session.add(
        SearchIndexState(
            id=1,
            desired_generation=1,
            active_generation=0,
            **{present: values[present]},
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_full_claim_allowed(db_session: Session) -> None:
    job = create_job(session=db_session, type=JobType.REBUILD_INDEX)
    db_session.flush()
    db_session.add(
        SearchIndexState(
            id=1,
            desired_generation=3,
            active_generation=0,
            rebuild_job_id=job.id,
            rebuild_target_generation=3,
            rebuild_claimed_at=_now(),
        )
    )
    db_session.flush()


def test_claim_target_cannot_exceed_desired(db_session: Session) -> None:
    job = create_job(session=db_session, type=JobType.REBUILD_INDEX)
    db_session.flush()
    db_session.add(
        SearchIndexState(
            id=1,
            desired_generation=2,
            active_generation=0,
            rebuild_job_id=job.id,
            rebuild_target_generation=5,
            rebuild_claimed_at=_now(),
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_claim_job_fk_restricts_job_deletion(db_session: Session) -> None:
    if db_session.bind is not None and db_session.bind.dialect.name != "postgresql":
        pytest.skip("FK RESTRICT enforcement requires PostgreSQL")

    job = create_job(session=db_session, type=JobType.REBUILD_INDEX)
    db_session.flush()
    create_search_index_state(
        session=db_session,
        desired_generation=1,
        active_generation=0,
        rebuild_job_id=job.id,
        rebuild_target_generation=1,
        rebuild_claimed_at=_now(),
    )

    db_session.delete(db_session.get(Job, job.id))
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()
