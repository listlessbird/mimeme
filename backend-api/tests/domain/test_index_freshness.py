from __future__ import annotations

import datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from tests.factories import create_job, create_search_index_state

from domain.index_freshness import (
    IndexFreshness,
    RebuildClaimOwnershipError,
    RebuildClaimTargetError,
    SearchIndexStateMissingError,
)
from shared.models import JobType


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _rebuild_job(session: Session) -> str:
    job = create_job(session=session, type=JobType.REBUILD_INDEX)
    session.flush()
    return job.id


# --- get / missing invariant -------------------------------------------------


def test_get_returns_seeded_view(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=1, active_generation=0)

    view = IndexFreshness(db_session).get()

    assert view.desired_generation == 1
    assert view.active_generation == 0
    assert view.is_stale is True
    assert view.active_claim is None


def test_missing_singleton_raises(db_session: Session) -> None:
    with pytest.raises(SearchIndexStateMissingError):
        IndexFreshness(db_session).get()


# --- mark_dirty --------------------------------------------------------------


def test_mark_dirty_increments_desired_and_preserves_active(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=4, active_generation=4)

    view = IndexFreshness(db_session).mark_dirty(reason="embedding_saved")

    assert view.desired_generation == 5
    assert view.active_generation == 4
    assert view.is_stale is True
    assert view.last_dirty_reason == "embedding_saved"
    assert view.last_dirty_at is not None


def test_mark_dirty_twice_advances_two_generations(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=1, active_generation=1)

    freshness = IndexFreshness(db_session)
    freshness.mark_dirty(reason="image_deleted")
    view = freshness.mark_dirty(reason="embedding_saved")

    assert view.desired_generation == 3
    assert view.active_generation == 1


def test_mark_dirty_missing_singleton_raises(db_session: Session) -> None:
    with pytest.raises(SearchIndexStateMissingError):
        IndexFreshness(db_session).mark_dirty(reason="embedding_saved")


# --- claim -------------------------------------------------------------------


def test_claim_dirty_acquires_then_second_is_busy(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=7, active_generation=3)
    job_a = _rebuild_job(db_session)
    job_b = _rebuild_job(db_session)
    freshness = IndexFreshness(db_session)

    first = freshness.claim(job_id=job_a, force=False, now=_now())
    second = freshness.claim(job_id=job_b, force=False, now=_now())

    assert first.acquired is True
    assert first.reason == "acquired"
    assert first.view.rebuild_job_id == job_a
    assert first.view.rebuild_target_generation == 7
    assert first.view.active_claim is not None

    assert second.acquired is False
    assert second.reason == "busy"
    assert second.view.rebuild_job_id == job_a


def test_claim_clean_non_force_returns_clean(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=5, active_generation=5)
    job = _rebuild_job(db_session)

    result = IndexFreshness(db_session).claim(job_id=job, force=False, now=_now())

    assert result.acquired is False
    assert result.reason == "clean"
    assert result.view.rebuild_job_id is None


def test_claim_clean_force_acquires(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=5, active_generation=5)
    job = _rebuild_job(db_session)

    result = IndexFreshness(db_session).claim(job_id=job, force=True, now=_now())

    assert result.acquired is True
    assert result.reason == "acquired"
    assert result.view.rebuild_target_generation == 5


def test_claim_force_does_not_steal_existing_claim(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=7, active_generation=3)
    owner = _rebuild_job(db_session)
    thief = _rebuild_job(db_session)
    freshness = IndexFreshness(db_session)
    freshness.claim(job_id=owner, force=False, now=_now())

    result = freshness.claim(job_id=thief, force=True, now=_now())

    assert result.acquired is False
    assert result.reason == "busy"
    assert result.view.rebuild_job_id == owner


def test_claim_captures_desired_under_lock(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=2, active_generation=0)
    job = _rebuild_job(db_session)
    freshness = IndexFreshness(db_session)

    result = freshness.claim(job_id=job, force=False, now=_now())
    view = freshness.mark_dirty(reason="embedding_saved")

    assert result.view.rebuild_target_generation == 2
    assert view.desired_generation == 3
    assert view.rebuild_target_generation == 2


# --- activate ----------------------------------------------------------------


def test_activate_advances_active_generation(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=6, active_generation=2)
    job = _rebuild_job(db_session)
    freshness = IndexFreshness(db_session)
    freshness.claim(job_id=job, force=False, now=_now())

    reconciled = _now()
    view = freshness.activate(job_id=job, target_generation=6, reconciled_at=reconciled)

    assert view.active_generation == 6
    assert view.is_stale is False
    assert view.last_reconciled_at is not None
    assert view.rebuild_job_id == job


def test_activate_wrong_owner_rejected(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=6, active_generation=2)
    owner = _rebuild_job(db_session)
    other = _rebuild_job(db_session)
    freshness = IndexFreshness(db_session)
    freshness.claim(job_id=owner, force=False, now=_now())

    with pytest.raises(RebuildClaimOwnershipError):
        freshness.activate(job_id=other, target_generation=6, reconciled_at=_now())


def test_activate_without_claim_rejected(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=6, active_generation=2)
    job = _rebuild_job(db_session)

    with pytest.raises(RebuildClaimOwnershipError):
        IndexFreshness(db_session).activate(job_id=job, target_generation=6, reconciled_at=_now())


def test_activate_wrong_target_rejected(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=6, active_generation=2)
    job = _rebuild_job(db_session)
    freshness = IndexFreshness(db_session)
    freshness.claim(job_id=job, force=False, now=_now())

    with pytest.raises(RebuildClaimTargetError):
        freshness.activate(job_id=job, target_generation=5, reconciled_at=_now())


def test_activate_change_during_build_stays_stale(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=10, active_generation=0)
    job = _rebuild_job(db_session)
    freshness = IndexFreshness(db_session)
    freshness.claim(job_id=job, force=False, now=_now())
    freshness.mark_dirty(reason="embedding_saved")

    view = freshness.activate(job_id=job, target_generation=10, reconciled_at=_now())

    assert view.desired_generation == 11
    assert view.active_generation == 10
    assert view.is_stale is True


# --- release -----------------------------------------------------------------


def test_release_clears_own_claim(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=6, active_generation=2)
    job = _rebuild_job(db_session)
    freshness = IndexFreshness(db_session)
    freshness.claim(job_id=job, force=False, now=_now())

    view = freshness.release(job_id=job)

    assert view.rebuild_job_id is None
    assert view.rebuild_target_generation is None
    assert view.rebuild_claimed_at is None


def test_release_no_claim_is_idempotent(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=6, active_generation=2)
    job = _rebuild_job(db_session)

    view = IndexFreshness(db_session).release(job_id=job)

    assert view.rebuild_job_id is None


def test_release_wrong_owner_rejected(db_session: Session) -> None:
    create_search_index_state(session=db_session, desired_generation=6, active_generation=2)
    owner = _rebuild_job(db_session)
    other = _rebuild_job(db_session)
    freshness = IndexFreshness(db_session)
    freshness.claim(job_id=owner, force=False, now=_now())

    with pytest.raises(RebuildClaimOwnershipError):
        freshness.release(job_id=other)

    assert IndexFreshness(db_session).get().rebuild_job_id == owner


# --- async caller through run_sync ------------------------------------------


async def test_async_caller_uses_module_through_run_sync(
    async_db_session: AsyncSession, run_sync_seed
) -> None:
    await run_sync_seed(
        lambda session: create_search_index_state(
            session=session, desired_generation=1, active_generation=1
        )
    )

    view = await async_db_session.run_sync(
        lambda sync_session: IndexFreshness(sync_session).mark_dirty(reason="image_deleted")
    )
    await async_db_session.flush()

    assert view.desired_generation == 2
    assert view.active_generation == 1
    assert view.last_dirty_reason == "image_deleted"
