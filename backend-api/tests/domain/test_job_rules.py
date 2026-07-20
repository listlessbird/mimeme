from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from domain.job_rules import (
    INGEST_URL_ERROR_LIMIT,
    JOB_ERROR_LIMIT,
    IngestJobResultPayload,
    JobLifecycleInvalidStateError,
    JobRowData,
    RawJobResultPayload,
    RebuildJobResultPayload,
    dedup_urls,
    derive_completion_status,
    ensure_cancellable,
    mint_ingest_job,
    mint_rebuild_job,
    parse_result,
    project_job,
    truncate_error,
)
from shared.models import JobStatus, JobType


@pytest.mark.parametrize(
    ("failed", "expected"),
    [
        (0, JobStatus.COMPLETED),
        (1, JobStatus.FAILED),
        (5, JobStatus.FAILED),
    ],
)
def test_ingest_completion_status_is_failed_when_any_item_failed(
    failed: int,
    expected: JobStatus,
) -> None:
    assert derive_completion_status(failed=failed) == expected


def test_ingest_result_payload_is_validated_and_serializable() -> None:
    payload = IngestJobResultPayload(processed=2, failed=0, duplicates=1)

    assert IngestJobResultPayload.model_validate_json(payload.model_dump_json()) == payload


def test_ingest_result_payload_rejects_negative_counts() -> None:
    with pytest.raises(ValidationError):
        IngestJobResultPayload(processed=1, failed=-1, duplicates=0)


def test_raw_result_payload_preserves_unparseable_legacy_storage_value() -> None:
    payload = RawJobResultPayload(raw="not-json")

    assert RawJobResultPayload.model_validate_json(payload.model_dump_json()) == payload


def test_rebuild_result_payload_is_validated_and_serializable() -> None:
    payload = RebuildJobResultPayload(
        version="v-1",
        num_vectors=10,
        dimension=768,
        removed_versions=["v-old"],
        text_num_vectors=9,
    )

    assert RebuildJobResultPayload.model_validate_json(payload.model_dump_json()) == payload


@pytest.mark.parametrize(
    "payload",
    [
        {
            "version": "v-1",
            "num_vectors": -1,
            "dimension": 768,
            "removed_versions": [],
            "text_num_vectors": None,
        },
        {
            "version": "v-1",
            "num_vectors": 1,
            "dimension": -1,
            "removed_versions": [],
            "text_num_vectors": None,
        },
        {
            "version": "v-1",
            "num_vectors": 1,
            "dimension": 768,
            "removed_versions": [],
            "text_num_vectors": -1,
        },
    ],
)
def test_rebuild_result_payload_rejects_invalid_counts(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RebuildJobResultPayload(**payload)


def test_mint_ingest_job_derives_workflow_id_from_job_id() -> None:
    job_id, workflow_id = mint_ingest_job()

    assert job_id.startswith("ingest-")
    assert len(job_id) == len("ingest-") + 12
    assert workflow_id == f"ingest-workflow-{job_id}"


def test_mint_rebuild_job_derives_workflow_id_from_job_id() -> None:
    job_id, workflow_id = mint_rebuild_job()

    assert job_id.startswith("rebuild-")
    assert len(job_id) == len("rebuild-") + 12
    assert workflow_id == f"rebuild-workflow-{job_id}"


def test_minted_job_ids_are_unique() -> None:
    assert mint_ingest_job()[0] != mint_ingest_job()[0]


def test_dedup_urls_preserves_order_and_counts_duplicates() -> None:
    unique_urls, duplicates = dedup_urls(["a", "b", "a", "c", "b", "a"])

    assert unique_urls == ["a", "b", "c"]
    assert duplicates == 3


def test_dedup_urls_empty_input() -> None:
    assert dedup_urls([]) == ([], 0)


@pytest.mark.parametrize("status", [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.CANCELLED])
def test_ensure_cancellable_allows_unfinished_jobs(status: JobStatus) -> None:
    ensure_cancellable(status)


@pytest.mark.parametrize("status", [JobStatus.COMPLETED, JobStatus.FAILED])
def test_ensure_cancellable_rejects_finished_jobs(status: JobStatus) -> None:
    with pytest.raises(JobLifecycleInvalidStateError):
        ensure_cancellable(status)


@pytest.mark.parametrize("limit", [JOB_ERROR_LIMIT, INGEST_URL_ERROR_LIMIT])
def test_truncate_error_caps_at_limit(limit: int) -> None:
    assert truncate_error("x" * (limit + 100), limit) == "x" * limit
    assert truncate_error("short", limit) == "short"
    assert truncate_error("", limit) == ""


def test_parse_result_none_is_none() -> None:
    assert parse_result(JobType.INGEST, None) is None


def test_parse_result_round_trips_typed_payloads() -> None:
    ingest = IngestJobResultPayload(processed=2, failed=1, duplicates=0)
    rebuild = RebuildJobResultPayload(
        version="v-1",
        num_vectors=10,
        dimension=768,
        removed_versions=[],
        text_num_vectors=None,
    )

    assert parse_result(JobType.INGEST, ingest.model_dump_json()) == ingest
    assert parse_result(JobType.REBUILD_INDEX, rebuild.model_dump_json()) == rebuild


def test_parse_result_preserves_malformed_json_as_raw() -> None:
    assert parse_result(JobType.INGEST, "not-json") == RawJobResultPayload(raw="not-json")
    assert parse_result(JobType.REBUILD_INDEX, "{}") == RawJobResultPayload(raw="{}")


def test_project_job_maps_row_and_parses_result() -> None:
    created = datetime(2026, 7, 13, tzinfo=UTC)
    payload = IngestJobResultPayload(processed=3, failed=0, duplicates=1)
    row = JobRowData(
        id="ingest-abc123def456",
        type=JobType.INGEST,
        status=JobStatus.COMPLETED,
        progress=100.0,
        message="done",
        created_at=created,
        started_at=created,
        completed_at=created,
        result=payload.model_dump_json(),
    )

    view = project_job(row)

    assert view.id == "ingest-abc123def456"
    assert view.status == JobStatus.COMPLETED
    assert view.result == payload
