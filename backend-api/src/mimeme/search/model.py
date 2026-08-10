from __future__ import annotations

import math
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Query(_Frozen):
    text: str | None = Field(default=None, max_length=200)
    similar_image_id: int | None = Field(default=None, gt=0)
    mode: Literal["image", "hybrid"] = "image"
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _one_input(self) -> Self:
        if self.text is not None:
            if not self.text.strip():
                raise ValueError("text must not be blank")
        if (self.text is None) == (self.similar_image_id is None):
            raise ValueError("provide exactly one of text or similar_image_id")
        if self.similar_image_id is not None and self.mode != "image":
            raise ValueError("similar search only supports image mode")
        return self


class Candidate(_Frozen):
    image_id: int = Field(gt=0)
    score: float

    @field_validator("score")
    @classmethod
    def _finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("candidate score must be finite")
        return value


class Batch(_Frozen):
    candidates: list[Candidate]
    cursor: str | None = None
    exhausted: bool
    version: str

    @model_validator(mode="after")
    def _unique_candidates(self) -> Self:
        ids = [candidate.image_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate image IDs must be unique within a batch")
        if self.exhausted and self.cursor is not None:
            raise ValueError("an exhausted batch cannot have a cursor")
        if not self.exhausted and self.cursor is None:
            raise ValueError("a non-exhausted batch requires a cursor")
        return self


class Result(_Frozen):
    id: int
    sha256: str
    score: float
    url: str | None = None
    caption: str | None = None
    ocr_text: str | None = None
    width: int | None = None
    height: int | None = None


class Page(_Frozen):
    query: str
    results: list[Result]
    total: int
    limit: int
    offset: int
    has_more: bool
    search_time_ms: float
    index_version: str | None


class Status(_Frozen):
    ready: bool
    serving_version: str | None = None
    candidate_version: str | None = None
    retained_version: str | None = None
    embed_model: str | None = None
    encoder_repo: str | None = None
    encoder_revision: str | None = None
    encoder_variant: str | None = None
    detail: str | None = None


class File(_Frozen):
    name: Literal[
        "index.faiss",
        "mapping.json",
        "metadata.json",
        "text_index.faiss",
        "text_mapping.json",
        "text_metadata.json",
    ]
    key: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Encoder(_Frozen):
    repo: str
    revision: str
    variant: str
    threads: int = Field(ge=1)


class Load(_Frozen):
    version: str = Field(min_length=1)
    files: list[File]
    encoder: Encoder
    hnsw_ef_search: int = Field(default=128, ge=1)

    @model_validator(mode="after")
    def _complete_generation(self) -> Self:
        names = [file.name for file in self.files]
        if len(names) != len(set(names)):
            raise ValueError("artifact names must be unique")
        required = {"index.faiss", "mapping.json", "metadata.json"}
        if not required.issubset(names):
            raise ValueError("image index generation is incomplete")
        text = {"text_index.faiss", "text_mapping.json", "text_metadata.json"}
        present = text.intersection(names)
        if present and present != text:
            raise ValueError("text index generation is incomplete")
        return self


class PreparedLoad(_Frozen):
    version: str
    paths: dict[str, str]
    encoder: Encoder
    hnsw_ef_search: int = Field(default=128, ge=1)


class Loaded(_Frozen):
    version: str
    embed_model: str
    dimension: int = Field(gt=0)
    image_count: int = Field(ge=0)
    text_count: int | None = Field(default=None, ge=0)
    faiss_version: str
    onnxruntime_version: str
    encoder_revision: str


class Switch(_Frozen):
    version: str


class Rollback(_Frozen):
    failed_version: str


class CandidateRequest(_Frozen):
    query: Query
    count: int = Field(ge=1, le=1000)
    cursor: str | None = None


class StatusCall(_Frozen):
    op: Literal["search.status"] = "search.status"


class QueryCall(_Frozen):
    op: Literal["search.query"] = "search.query"
    request: CandidateRequest


class LoadCall(_Frozen):
    op: Literal["search.load"] = "search.load"
    load: PreparedLoad


class SwitchCall(_Frozen):
    op: Literal["search.switch"] = "search.switch"
    version: str


class RollbackCall(_Frozen):
    op: Literal["search.rollback"] = "search.rollback"
    failed_version: str


ChildCall = Annotated[
    StatusCall | QueryCall | LoadCall | SwitchCall | RollbackCall,
    Field(discriminator="op"),
]
