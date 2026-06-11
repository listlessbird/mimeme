from __future__ import annotations

from pydantic import BaseModel

from shared.models import DuplicateReason, ProcessingStatus, SourceRunStatus


class UrlOutcome(BaseModel, frozen=True):
    status: ProcessingStatus
    duplicate_reason: DuplicateReason | None = None


class RunAccounting(BaseModel, frozen=True):
    status: SourceRunStatus
    discovered: int
    queued: int
    duplicate: int
    failed: int


def derive_run_accounting(
    *,
    discovered_items: int,
    url_outcomes: list[UrlOutcome],
) -> RunAccounting:
    queued = len(url_outcomes)
    duplicate = sum(1 for outcome in url_outcomes if outcome.duplicate_reason is not None)
    failed = sum(1 for outcome in url_outcomes if outcome.status == ProcessingStatus.FAILED)

    if queued == 0:
        status = SourceRunStatus.COMPLETED
    elif failed == 0:
        status = SourceRunStatus.COMPLETED
    elif failed == queued:
        status = SourceRunStatus.FAILED
    else:
        status = SourceRunStatus.PARTIAL

    return RunAccounting(
        status=status,
        discovered=discovered_items,
        queued=queued,
        duplicate=duplicate,
        failed=failed,
    )
