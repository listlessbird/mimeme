from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import func, update
from sqlalchemy.orm import Session

from shared.models import SearchIndexState

SINGLETON_ID = 1


class SearchIndexStateMissingError(Exception):
    pass


class RebuildClaimOwnershipError(Exception):
    pass


class RebuildClaimTargetError(Exception):
    pass


class RebuildClaim(BaseModel, frozen=True):
    job_id: str
    target_generation: int
    claimed_at: datetime


class IndexFreshnessView(BaseModel, frozen=True):
    desired_generation: int
    active_generation: int
    is_stale: bool
    rebuild_job_id: str | None
    rebuild_target_generation: int | None
    rebuild_claimed_at: datetime | None
    last_dirty_at: datetime | None
    last_dirty_reason: str | None
    last_reconciled_at: datetime | None

    @property
    def active_claim(self) -> RebuildClaim | None:
        if (
            self.rebuild_job_id is None
            or self.rebuild_target_generation is None
            or self.rebuild_claimed_at is None
        ):
            return None
        return RebuildClaim(
            job_id=self.rebuild_job_id,
            target_generation=self.rebuild_target_generation,
            claimed_at=self.rebuild_claimed_at,
        )


class ClaimResult(BaseModel, frozen=True):
    acquired: bool
    reason: Literal["acquired", "clean", "busy"]
    view: IndexFreshnessView


def _to_view(row: SearchIndexState) -> IndexFreshnessView:
    return IndexFreshnessView(
        desired_generation=row.desired_generation,
        active_generation=row.active_generation,
        is_stale=row.desired_generation > row.active_generation,
        rebuild_job_id=row.rebuild_job_id,
        rebuild_target_generation=row.rebuild_target_generation,
        rebuild_claimed_at=row.rebuild_claimed_at,
        last_dirty_at=row.last_dirty_at,
        last_dirty_reason=row.last_dirty_reason,
        last_reconciled_at=row.last_reconciled_at,
    )


class IndexFreshness:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self) -> IndexFreshnessView:
        return _to_view(self._require())

    def lock(self) -> IndexFreshnessView:
        return _to_view(self._require(lock=True))

    def mark_dirty(self, *, reason: str) -> IndexFreshnessView:
        stmt = (
            update(SearchIndexState)
            .where(SearchIndexState.id == SINGLETON_ID)
            .values(
                desired_generation=SearchIndexState.desired_generation + 1,
                last_dirty_at=func.now(),
                last_dirty_reason=reason,
            )
            .returning(SearchIndexState)
            .execution_options(populate_existing=True)
        )
        row = self._session.execute(stmt).scalars().one_or_none()
        if row is None:
            raise SearchIndexStateMissingError("search_index_state singleton row is missing")
        return _to_view(row)

    def claim(self, *, job_id: str, force: bool, now: datetime) -> ClaimResult:
        row = self._require(lock=True)

        if row.rebuild_job_id is not None:
            return ClaimResult(acquired=False, reason="busy", view=_to_view(row))

        if not force and row.desired_generation <= row.active_generation:
            return ClaimResult(acquired=False, reason="clean", view=_to_view(row))

        row.rebuild_job_id = job_id
        row.rebuild_target_generation = row.desired_generation
        row.rebuild_claimed_at = now
        self._session.flush()
        return ClaimResult(acquired=True, reason="acquired", view=_to_view(row))

    def activate(
        self, *, job_id: str, target_generation: int, reconciled_at: datetime
    ) -> IndexFreshnessView:
        row = self._require(lock=True)

        if row.rebuild_job_id != job_id:
            raise RebuildClaimOwnershipError(
                f"Job {job_id} does not own the rebuild claim (owner={row.rebuild_job_id})"
            )
        if row.rebuild_target_generation != target_generation:
            raise RebuildClaimTargetError(
                f"activation target {target_generation} does not match claim target "
                f"{row.rebuild_target_generation}"
            )

        row.active_generation = target_generation
        row.last_reconciled_at = reconciled_at
        self._session.flush()
        return _to_view(row)

    def release(self, *, job_id: str) -> IndexFreshnessView:
        row = self._require(lock=True)

        if row.rebuild_job_id is None:
            return _to_view(row)
        if row.rebuild_job_id != job_id:
            raise RebuildClaimOwnershipError(
                f"Job {job_id} cannot release a claim owned by {row.rebuild_job_id}"
            )

        row.rebuild_job_id = None
        row.rebuild_target_generation = None
        row.rebuild_claimed_at = None
        self._session.flush()
        return _to_view(row)

    def _require(self, *, lock: bool = False) -> SearchIndexState:
        row = self._session.get(SearchIndexState, SINGLETON_ID, with_for_update=lock or None)
        if row is None:
            raise SearchIndexStateMissingError("search_index_state singleton row is missing")
        return row
