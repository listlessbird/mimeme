from __future__ import annotations

import pytest
from pydantic import ValidationError

from domain.job_rules import (
    IngestJobResultPayload,
    RawJobResultPayload,
    RebuildJobResultPayload,
    derive_completion_status,
)
from shared.models import JobStatus


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
            "dimension": 0,
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
