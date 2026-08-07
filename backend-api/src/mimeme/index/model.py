from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Trigger(StrEnum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"


class Phase(StrEnum):
    NEW = "new"
    PREPARED = "prepared"
    BUILT = "built"
    ACTIVE = "active"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RELEASED = "released"


class State(_Frozen):
    job_id: str = Field(min_length=1)
    phase: Phase = Phase.NEW


class Embedding(_Frozen):
    image_id: int = Field(gt=0)
    image_key: str | None = Field(default=None, min_length=1)
    text_key: str | None = Field(default=None, min_length=1)
    shard: int | None = Field(default=None, ge=0)
    row: int | None = Field(default=None, ge=0)
    text_present: bool = False

    @property
    def sealed(self) -> bool:
        return self.shard is not None

    @model_validator(mode="after")
    def _one_layout(self) -> Self:
        if (self.shard is None) != (self.row is None):
            raise ValueError("a shard coordinate needs both a shard and a row")
        if (self.image_key is None) == (self.shard is None):
            raise ValueError("an embedding is either sealed into a shard or a standalone object")
        if self.text_key is not None and self.image_key is None:
            raise ValueError("a sealed embedding carries no standalone text object")
        if self.text_present and self.shard is None:
            raise ValueError("text presence on a standalone embedding is its text key")
        return self


class Encoder(_Frozen):
    repo: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    variant: str = Field(min_length=1)
    threads: int = Field(default=1, ge=1)


class Build(_Frozen):
    job_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    target_generation: int = Field(ge=0)
    model: str = Field(min_length=1)
    index_type: Literal["flat", "hnsw"]
    dimension: int = Field(ge=0)
    native_threads: int = Field(default=1, ge=1)
    encoder: Encoder
    embeddings: list[Embedding]
    planned_reads: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _unique_images(self) -> Self:
        ids = [item.image_id for item in self.embeddings]
        if len(ids) != len(set(ids)):
            raise ValueError("embedding image IDs must be unique")
        if self.embeddings and self.dimension == 0:
            raise ValueError("a non-empty build requires an embedding dimension")
        return self


ArtifactName = Literal[
    "index.faiss",
    "mapping.json",
    "metadata.json",
    "text_index.faiss",
    "text_mapping.json",
    "text_metadata.json",
]


class File(_Frozen):
    name: ArtifactName
    key: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    length: int = Field(ge=0)


class Manifest(_Frozen):
    format_version: Literal[1] = 1
    version: str = Field(min_length=1)
    target_generation: int = Field(ge=0)
    model: str = Field(min_length=1)
    index_type: Literal["flat", "hnsw"]
    encoder: Encoder
    dimension: int = Field(ge=0)
    image_count: int = Field(ge=0)
    text_count: int | None = Field(default=None, ge=0)
    files: list[File]
    complete_key: str = Field(min_length=1)

    @model_validator(mode="after")
    def _complete(self) -> Self:
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
        prefix = f"indexes/{self.version}/"
        if self.complete_key != f"{prefix}complete.json":
            raise ValueError("completeness manifest must use its generation prefix")
        if any(not file.key.startswith(prefix) for file in self.files):
            raise ValueError("every artifact must use its generation prefix")
        return self


class Result(_Frozen):
    outcome: Literal["built", "empty"]
    manifest: Manifest | None = None

    @model_validator(mode="after")
    def _manifest_matches_outcome(self) -> Self:
        if (self.outcome == "built") != (self.manifest is not None):
            raise ValueError("built results require a manifest and empty results forbid one")
        return self


class PrepareInput(_Frozen):
    job_id: str | None
    workflow_id: str
    force: bool = False
    trigger: Trigger
    model: str
    index_type: Literal["flat", "hnsw"]


class Prepared(_Frozen):
    decision: Literal["build", "clean", "busy", "deferred"]
    job_id: str | None = None
    build: Build | None = None

    @model_validator(mode="after")
    def _build_decision_has_manifest(self) -> Self:
        if (self.decision == "build") != (self.build is not None):
            raise ValueError("only a build decision carries a build manifest")
        return self


class ActivateInput(_Frozen):
    job_id: str
    target_generation: int = Field(ge=0)
    result: Result | None = None
    error: str | None = None
    cancelled: bool = False

    @model_validator(mode="after")
    def _one_outcome(self) -> Self:
        outcomes = int(self.result is not None) + int(self.error is not None) + int(self.cancelled)
        if outcomes != 1:
            raise ValueError("activation requires exactly one terminal outcome")
        return self


class Activated(_Frozen):
    version: str
    removed_versions: list[str] = []


class Backfilled(_Frozen):
    model: str
    text_objects: int = Field(ge=0)
    marked_present: int = Field(ge=0)
    marked_absent: int = Field(ge=0)


class Snapshot(_Frozen):
    target_generation: int = Field(ge=0)
    dimension: int = Field(ge=0)
    embeddings: list[Embedding]


class WorkflowInput(_Frozen):
    job_id: str | None
    force: bool = False
    model: str
    index_type: Literal["flat", "hnsw"]
    trigger: Trigger
    busy_attempt: int = Field(default=0, ge=0)


class WorkflowResult(_Frozen):
    job_id: str | None
    outcome: Literal["built", "empty", "clean", "busy", "deferred"]
    version: str | None = None


class LocalShard(_Frozen):
    number: int = Field(ge=0)
    image_path: str
    text_path: str | None = None


class LocalEmbedding(_Frozen):
    image_id: int = Field(gt=0)
    image_path: str | None = None
    text_path: str | None = None
    shard: int | None = Field(default=None, ge=0)
    row: int | None = Field(default=None, ge=0)
    text_present: bool = False


class PreparedBuild(_Frozen):
    version: str
    target_generation: int = Field(ge=0)
    model: str
    index_type: Literal["flat", "hnsw"]
    dimension: int = Field(ge=0)
    native_threads: int = Field(default=1, ge=1)
    encoder: Encoder
    output_dir: str
    shards: list[LocalShard] = []
    embeddings: list[LocalEmbedding]


class BuildCall(_Frozen):
    op: Literal["index.build"] = "index.build"
    build: PreparedBuild


class LocalMember(_Frozen):
    image_id: int = Field(gt=0)
    image_path: str
    text_path: str | None = None


class PackCall(_Frozen):
    op: Literal["index.pack"] = "index.pack"
    members: list[LocalMember]
    image_out: str
    text_out: str


class PackedFile(_Frozen):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    length: int = Field(ge=0)


class Packed(_Frozen):
    rows: int = Field(ge=0)
    dimension: int = Field(gt=0)
    image: PackedFile
    text: PackedFile


IndexCall = Annotated[BuildCall | PackCall, Field(discriminator="op")]


class BuiltFile(_Frozen):
    name: ArtifactName
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    length: int = Field(ge=0)


class Built(_Frozen):
    version: str
    target_generation: int = Field(ge=0)
    model: str
    index_type: Literal["flat", "hnsw"]
    dimension: int = Field(ge=0)
    image_count: int = Field(ge=0)
    text_count: int | None = Field(default=None, ge=0)
    files: list[BuiltFile]


class BuildSpec(_Frozen):
    op: Literal["index.build"] = "index.build"
    build: Build


class SealMember(_Frozen):
    image_id: int = Field(gt=0)
    image_key: str = Field(min_length=1)
    text_key: str | None = Field(default=None, min_length=1)


class SealShard(_Frozen):
    number: int = Field(ge=0)
    image_key: str = Field(min_length=1)
    text_key: str = Field(min_length=1)
    members: list[SealMember]

    @model_validator(mode="after")
    def _has_members(self) -> Self:
        if not self.members:
            raise ValueError("a shard needs at least one member")
        return self


class Seal(_Frozen):
    job_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    shards: list[SealShard]


class SealSpec(_Frozen):
    op: Literal["index.seal"] = "index.seal"
    seal: Seal


class SealInput(_Frozen):
    job_id: str = Field(min_length=1)
    model: str = Field(min_length=1)


class Sealed(_Frozen):
    model: str = Field(min_length=1)
    shards: int = Field(ge=0)
    rows: int = Field(ge=0)


class SealedShard(_Frozen):
    number: int = Field(ge=0)
    rows: int = Field(ge=0)


class SealResult(_Frozen):
    shards: list[SealedShard] = []
    error: str | None = None
