from __future__ import annotations

import pytest

from mimeme.db.schema import DuplicateReason, ProcessingStatus, SourceRunStatus
from mimeme.domain.source_run_accounting import (
    RunAccounting,
    UrlOutcome,
    derive_run_accounting,
)


def _done(reason: DuplicateReason | None = None) -> UrlOutcome:
    return UrlOutcome(status=ProcessingStatus.DONE, duplicate_reason=reason)


def _failed() -> UrlOutcome:
    return UrlOutcome(status=ProcessingStatus.FAILED)


# --- counts ---------------------------------------------------------------


def test_empty_poll_is_completed_with_zero_counts() -> None:
    accounting = derive_run_accounting(discovered_items=0, url_outcomes=[])

    assert accounting == RunAccounting(
        status=SourceRunStatus.COMPLETED,
        discovered=0,
        queued=0,
        duplicate=0,
        failed=0,
    )


def test_all_already_seen_is_completed_and_discovered_passes_through() -> None:
    # Five items discovered, none new (all dropped pre-download as SOURCE_SEEN),
    # so nothing was queued. A healthy "nothing new" poll, not a failure.
    accounting = derive_run_accounting(discovered_items=5, url_outcomes=[])

    assert accounting.status == SourceRunStatus.COMPLETED
    assert accounting.discovered == 5
    assert accounting.queued == 0


def test_queued_counts_every_url_outcome() -> None:
    accounting = derive_run_accounting(
        discovered_items=3,
        url_outcomes=[_done(), _done(), _failed()],
    )

    assert accounting.queued == 3


def test_duplicate_counts_outcomes_with_a_reason_regardless_of_which() -> None:
    accounting = derive_run_accounting(
        discovered_items=3,
        url_outcomes=[
            _done(DuplicateReason.SOURCE_SEEN),
            _done(DuplicateReason.SHA256),
            _done(DuplicateReason.PHASH),
        ],
    )

    assert accounting.duplicate == 3
    assert accounting.failed == 0


def test_failed_counts_by_status_not_by_duplicate_reason() -> None:
    # A FAILED url with no dedup reason is a failure, never a duplicate.
    accounting = derive_run_accounting(
        discovered_items=2,
        url_outcomes=[_failed(), _done(DuplicateReason.SHA256)],
    )

    assert accounting.failed == 1
    assert accounting.duplicate == 1


def test_mixed_run_counts_are_independent() -> None:
    accounting = derive_run_accounting(
        discovered_items=10,
        url_outcomes=[
            _done(),
            _done(),
            _done(DuplicateReason.PHASH),
            _failed(),
        ],
    )

    assert accounting.discovered == 10
    assert accounting.queued == 4
    assert accounting.duplicate == 1
    assert accounting.failed == 1


# --- status rule ----------------------------------------------------------


def test_no_failures_with_queued_work_is_completed() -> None:
    accounting = derive_run_accounting(
        discovered_items=2,
        url_outcomes=[_done(), _done(DuplicateReason.SHA256)],
    )

    assert accounting.status == SourceRunStatus.COMPLETED


def test_some_failed_some_succeeded_is_partial() -> None:
    accounting = derive_run_accounting(
        discovered_items=2,
        url_outcomes=[_done(), _failed()],
    )

    assert accounting.status == SourceRunStatus.PARTIAL


def test_all_queued_work_failed_is_failed() -> None:
    accounting = derive_run_accounting(
        discovered_items=2,
        url_outcomes=[_failed(), _failed()],
    )

    assert accounting.status == SourceRunStatus.FAILED


@pytest.mark.parametrize(
    ("done", "failed", "expected"),
    [
        (0, 0, SourceRunStatus.COMPLETED),  # queued==0
        (3, 0, SourceRunStatus.COMPLETED),  # all ok
        (2, 1, SourceRunStatus.PARTIAL),  # mix
        (0, 3, SourceRunStatus.FAILED),  # all failed
        (1, 0, SourceRunStatus.COMPLETED),  # single ok
        (0, 1, SourceRunStatus.FAILED),  # single failed == all failed
    ],
)
def test_status_rule_boundaries(done: int, failed: int, expected: SourceRunStatus) -> None:
    outcomes = [_done() for _ in range(done)] + [_failed() for _ in range(failed)]

    accounting = derive_run_accounting(discovered_items=done + failed, url_outcomes=outcomes)

    assert accounting.status == expected


def test_result_is_frozen() -> None:
    accounting = derive_run_accounting(discovered_items=0, url_outcomes=[])

    with pytest.raises((AttributeError, TypeError, ValueError)):
        accounting.status = SourceRunStatus.FAILED  # type: ignore[misc]
