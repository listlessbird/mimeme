from __future__ import annotations

from pydantic import BaseModel, Field

from shared.models import JobStatus


class IngestJobResultPayload(BaseModel, frozen=True):
    processed: int = Field(ge=0)
    failed: int = Field(ge=0)
    duplicates: int = Field(ge=0)


class RebuildJobResultPayload(BaseModel, frozen=True):
    version: str
    num_vectors: int = Field(ge=0)
    dimension: int = Field(ge=1)
    removed_versions: list[str]
    text_num_vectors: int | None = Field(default=None, ge=0)


class RawJobResultPayload(BaseModel, frozen=True):
    raw: str


type JobResultPayload = IngestJobResultPayload | RebuildJobResultPayload | RawJobResultPayload


def derive_completion_status(*, failed: int) -> JobStatus:
    return JobStatus.COMPLETED if failed == 0 else JobStatus.FAILED
