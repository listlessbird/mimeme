from __future__ import annotations

import pytest

from mimeme.db.schema import JobStatus, JobType
from mimeme.job import rule
from mimeme.job.model import (
    IngestResult,
    InvalidState,
    RawResult,
    RebuildResult,
    RowData,
)


def _row(*, job_type: JobType = JobType.INGEST, result: str | None = None) -> RowData:
    from datetime import UTC, datetime

    return RowData(
        id="job-1",
        type=job_type,
        status=JobStatus.COMPLETED,
        progress=100.0,
        message=None,
        created_at=datetime.now(UTC),
        started_at=None,
        completed_at=None,
        result=result,
    )


class TestMint:
    def test_ingest_ids_are_prefixed_and_paired(self) -> None:
        job_id, workflow_id = rule.mint_ingest()
        assert job_id.startswith("ingest-")
        assert workflow_id == f"ingest-workflow-{job_id}"

    def test_rebuild_ids_are_prefixed_and_paired(self) -> None:
        job_id, workflow_id = rule.mint_rebuild()
        assert job_id.startswith("rebuild-")
        assert workflow_id == f"rebuild-workflow-{job_id}"

    def test_ids_are_unique(self) -> None:
        assert rule.mint_ingest()[0] != rule.mint_ingest()[0]


class TestCancellable:
    @pytest.mark.parametrize(
        "status",
        [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.CANCELLED],
    )
    def test_non_terminal_and_cancelled_are_cancellable(self, status: JobStatus) -> None:
        rule.ensure_cancellable(status)

    @pytest.mark.parametrize("status", [JobStatus.COMPLETED, JobStatus.FAILED])
    def test_terminal_is_not_cancellable(self, status: JobStatus) -> None:
        with pytest.raises(InvalidState):
            rule.ensure_cancellable(status)


class TestCompletionStatus:
    def test_no_failures_completes(self) -> None:
        assert rule.derive_completion_status(failed=0) is JobStatus.COMPLETED

    @pytest.mark.parametrize("failed", [1, 5, 100])
    def test_any_failure_fails(self, failed: int) -> None:
        assert rule.derive_completion_status(failed=failed) is JobStatus.FAILED


class TestTruncate:
    def test_within_limit_is_unchanged(self) -> None:
        assert rule.truncate("short", 1000) == "short"

    def test_over_limit_is_cut(self) -> None:
        assert rule.truncate("x" * 5000, 2000) == "x" * 2000


class TestParseResult:
    def test_none_result_is_none(self) -> None:
        assert rule.parse_result(JobType.INGEST, None) is None

    def test_ingest_result_round_trips(self) -> None:
        payload = IngestResult(processed=5, failed=0, duplicates=1).model_dump_json()
        parsed = rule.parse_result(JobType.INGEST, payload)
        assert parsed == IngestResult(processed=5, failed=0, duplicates=1)

    def test_rebuild_result_round_trips(self) -> None:
        payload = RebuildResult(
            version="v-1",
            num_vectors=10,
            dimension=768,
            removed_versions=["v-old"],
            text_num_vectors=9,
        ).model_dump_json()
        parsed = rule.parse_result(JobType.REBUILD_INDEX, payload)
        assert isinstance(parsed, RebuildResult)
        assert parsed.version == "v-1"

    def test_malformed_json_falls_back_to_raw(self) -> None:
        parsed = rule.parse_result(JobType.INGEST, "not-json")
        assert parsed == RawResult(raw="not-json")

    def test_ingest_type_never_parses_as_rebuild(self) -> None:
        payload = RebuildResult(
            version="v", num_vectors=1, dimension=1, removed_versions=[]
        ).model_dump_json()
        parsed = rule.parse_result(JobType.INGEST, payload)
        assert isinstance(parsed, RawResult)


class TestProject:
    def test_projects_row_with_parsed_result(self) -> None:
        row = _row(result=IngestResult(processed=1, failed=0, duplicates=0).model_dump_json())
        view = rule.project(row)
        assert view.id == "job-1"
        assert isinstance(view.result, IngestResult)

    def test_projects_row_without_result(self) -> None:
        assert rule.project(_row()).result is None
