from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from mimeme.db.schema import IndexBuild, Job, JobStatus, JobType, RebuildTrigger
from mimeme.domain.index_freshness import IndexFreshness
from mimeme.domain.job_lifecycle import JobLifecycle
from mimeme.domain.job_rules import mint_rebuild_job

_TERMINAL = (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


class PrepareDecision(BaseModel, frozen=True):
    decision: Literal["build", "clean", "busy"]
    job_id: str | None = None
    target_generation: int | None = None


class RebuildCoordinator:
    def __init__(self, session: Session) -> None:
        self._session = session
        self._freshness = IndexFreshness(session)
        self._jobs = JobLifecycle(session)

    def prepare(
        self,
        *,
        job_id: str | None,
        workflow_id: str,
        force: bool,
        trigger: RebuildTrigger,
        now: datetime,
        claim_timeout: timedelta,
    ) -> PrepareDecision:
        view = self._freshness.lock()

        claim = view.active_claim
        if claim is not None:
            owner = self._session.get(Job, claim.job_id)
            if owner is None or owner.status in _TERMINAL:
                self._freshness.release(job_id=claim.job_id)
                view = self._freshness.lock()
            elif claim.claimed_at + claim_timeout <= now:
                self._jobs.fail_rebuild_job(
                    claim.job_id, f"Rebuild claim expired; reclaimed by {trigger.value}"
                )
                self._freshness.release(job_id=claim.job_id)
                view = self._freshness.lock()

        if view.active_claim is not None:
            return PrepareDecision(
                decision="busy",
                job_id=job_id if trigger is RebuildTrigger.MANUAL else None,
            )

        if not view.is_stale and not force:
            if trigger is RebuildTrigger.MANUAL and job_id is not None:
                active = self._active_build()
                self._jobs.complete_rebuild_job(
                    job_id=job_id,
                    version=active.version if active else "",
                    num_vectors=(active.num_vectors if active else None) or 0,
                    dimension=(active.dimension if active else None) or 0,
                    removed_versions=[],
                    text_num_vectors=None,
                    skipped=True,
                    skip_reason="already_current",
                    message="Index already current",
                )
            return PrepareDecision(
                decision="clean",
                job_id=job_id if trigger is RebuildTrigger.MANUAL else None,
            )

        if trigger is RebuildTrigger.SCHEDULED:
            job_id = self._create_scheduled_job(workflow_id)

        assert job_id is not None
        result = self._freshness.claim(job_id=job_id, force=force, now=now)
        return PrepareDecision(
            decision="build",
            job_id=job_id,
            target_generation=result.view.rebuild_target_generation,
        )

    def _create_scheduled_job(self, workflow_id: str) -> str:
        job_id, _ = mint_rebuild_job()
        self._session.add(Job(id=job_id, type=JobType.REBUILD_INDEX, workflow_id=workflow_id))
        self._session.flush()
        return job_id

    def _active_build(self) -> IndexBuild | None:
        return self._session.scalars(
            select(IndexBuild).where(IndexBuild.is_active.is_(True))
        ).first()
