from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mimeme.search.document import SearchDocument


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
    seq: int | None = Field(default=None, ge=0)
    text_present: bool = False

    @property
    def sealed(self) -> bool:
        return self.shard is not None

    @model_validator(mode="after")
    def _one_layout(self) -> Self:
        if (self.shard is None) != (self.row is None):
            raise ValueError("a shard coordinate needs both a shard and a row")
        if (self.shard is None) != (self.seq is None):
            raise ValueError("a shard coordinate needs the object generation it lives in")
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


class DocumentFile(_Frozen):
    name: Literal["documents.jsonl.zst"] = "documents.jsonl.zst"
    key: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    length: int = Field(ge=0)
    count: int = Field(ge=0)
    projection_version: Literal[1] = 1


class Bm25File(_Frozen):
    name: Literal["bm25.sqlite3"] = "bm25.sqlite3"
    key: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    length: int = Field(gt=0)
    count: int = Field(ge=0)
    schema_version: Literal[1] = 1
    projection_version: Literal[1] = 1
    tokenizer: Literal["porter unicode61"] = "porter unicode61"
    weights: tuple[float, float, float, float, float, float, float]
    sqlite_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")

    @model_validator(mode="after")
    def _supported_weights(self) -> Self:
        if self.weights != (4, 4, 4, 2, 2, 2, 1):
            raise ValueError("BM25 field weights are incompatible")
        return self


class Blob(_Frozen):
    name: str = Field(min_length=1)
    key: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    length: int = Field(gt=0)


class DenseVectors(_Frozen):
    retriever: Literal["bge"] = "bge"
    version: str = Field(min_length=1)
    model: Literal["BAAI/bge-small-en-v1.5"] = "BAAI/bge-small-en-v1.5"
    dimension: Literal[384] = 384
    encoder: Encoder
    document_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    projection_version: Literal[1] = 1
    render_version: Literal[1] = 1
    count: int = Field(ge=0)
    vectors: Blob
    mapping: Blob
    metadata: Blob

    @model_validator(mode="after")
    def _owned_by_generation(self) -> Self:
        prefix = f"indexes/{self.version}/"
        names = {
            self.vectors.name: "bge_vectors.npy",
            self.mapping.name: "bge_vectors_mapping.json",
            self.metadata.name: "bge_vectors_metadata.json",
        }
        if names != {
            "bge_vectors.npy": "bge_vectors.npy",
            "bge_vectors_mapping.json": "bge_vectors_mapping.json",
            "bge_vectors_metadata.json": "bge_vectors_metadata.json",
        }:
            raise ValueError("BGE vector artifact names are incompatible")
        if any(blob.key != f"{prefix}{blob.name}" for blob in self.blobs):
            raise ValueError("BGE vector artifacts must use their generation prefix")
        return self

    @property
    def blobs(self) -> tuple[Blob, Blob, Blob]:
        return (self.vectors, self.mapping, self.metadata)


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
    documents: DocumentFile | None = None
    bm25: Bm25File | None = None
    dense_vectors: list[DenseVectors] = []
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
    "bge_index.faiss",
    "bge_mapping.json",
    "bge_metadata.json",
]


class File(_Frozen):
    name: ArtifactName
    key: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    length: int = Field(ge=0)


class DenseIndex(_Frozen):
    retriever: Literal["bge"] = "bge"
    model: Literal["BAAI/bge-small-en-v1.5"] = "BAAI/bge-small-en-v1.5"
    dimension: Literal[384] = 384
    encoder: Encoder
    document_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    projection_version: Literal[1] = 1
    render_version: Literal[1] = 1
    count: int = Field(ge=0)
    files: tuple[File, File, File]

    @model_validator(mode="after")
    def _complete(self) -> Self:
        if {file.name for file in self.files} != {
            "bge_index.faiss",
            "bge_mapping.json",
            "bge_metadata.json",
        }:
            raise ValueError("BGE dense index is incomplete")
        return self


class Manifest(_Frozen):
    format_version: Literal[1, 2] = 1
    version: str = Field(min_length=1)
    target_generation: int = Field(ge=0)
    model: str = Field(min_length=1)
    index_type: Literal["flat", "hnsw"]
    encoder: Encoder
    dimension: int = Field(ge=0)
    image_count: int = Field(ge=0)
    text_count: int | None = Field(default=None, ge=0)
    files: list[File]
    documents: DocumentFile | None = None
    bm25: Bm25File | None = None
    dense_vectors: list[DenseVectors] = []
    dense: list[DenseIndex] = []
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
        if self.format_version == 1 and self.documents is not None:
            raise ValueError("manifest v1 cannot contain a document artifact")
        if self.format_version == 2 and self.documents is None:
            raise ValueError("manifest v2 requires a document artifact")
        if self.documents is not None and self.documents.key != f"{prefix}{self.documents.name}":
            raise ValueError("document artifact must use its generation prefix")
        if self.documents is not None and self.documents.count != self.image_count:
            raise ValueError("document count must match image count")
        if self.bm25 is not None and self.bm25.key != f"{prefix}{self.bm25.name}":
            raise ValueError("BM25 artifact must use its generation prefix")
        if self.bm25 is not None and self.bm25.count != self.image_count:
            raise ValueError("BM25 document count must match image count")
        if self.bm25 is not None and self.documents is None:
            raise ValueError("BM25 requires the canonical document artifact")
        if (
            self.bm25 is not None
            and self.documents is not None
            and self.bm25.projection_version != self.documents.projection_version
        ):
            raise ValueError("BM25 and document projection versions must match")
        dense_retrievers = [item.retriever for item in self.dense]
        if len(dense_retrievers) != len(set(dense_retrievers)):
            raise ValueError("dense retriever IDs must be unique")
        file_by_name = {file.name: file for file in self.files}
        for item in self.dense:
            if self.documents is None:
                raise ValueError("dense indexes require the canonical document artifact")
            if item.count != self.documents.count:
                raise ValueError("dense index count must match document count")
            if item.document_content_sha256 != self.documents.content_sha256:
                raise ValueError("dense index document checksum does not match the snapshot")
            if item.projection_version != self.documents.projection_version:
                raise ValueError("dense index projection does not match the snapshot")
            if any(file_by_name.get(file.name) != file for file in item.files):
                raise ValueError("dense index files must belong to the generation file set")
        vector_retrievers = [item.retriever for item in self.dense_vectors]
        if vector_retrievers != dense_retrievers:
            raise ValueError("dense vector and index retrievers must match")
        for vectors, dense_index in zip(self.dense_vectors, self.dense, strict=True):
            if vectors.version != self.version:
                raise ValueError("dense vectors do not belong to this generation")
            if (
                vectors.model != dense_index.model
                or vectors.dimension != dense_index.dimension
                or vectors.encoder != dense_index.encoder
                or vectors.document_content_sha256 != dense_index.document_content_sha256
                or vectors.projection_version != dense_index.projection_version
                or vectors.render_version != dense_index.render_version
                or vectors.count != dense_index.count
            ):
                raise ValueError("dense vector and index identities do not match")
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


class EmbeddingManifest(_Frozen):
    version: str = Field(min_length=1)
    dimension: int = Field(ge=0)
    embeddings: list[Embedding]


class BuildPlan(_Frozen):
    job_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    target_generation: int = Field(ge=0)
    model: str = Field(min_length=1)
    index_type: Literal["flat", "hnsw"]
    dimension: int = Field(ge=0)
    native_threads: int = Field(default=1, ge=1)
    encoder: Encoder
    embeddings_key: str = Field(min_length=1)
    documents: DocumentFile | None = None
    bm25: Bm25File | None = None
    dense_vectors: list[DenseVectors] = []
    num_embeddings: int = Field(ge=0)
    planned_reads: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _sized_plan(self) -> Self:
        if self.num_embeddings and self.dimension == 0:
            raise ValueError("a non-empty build requires an embedding dimension")
        return self


class Prepared(_Frozen):
    decision: Literal["build", "clean", "busy", "deferred"]
    job_id: str | None = None
    build: BuildPlan | None = None

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
    documents: list[SearchDocument]

    @model_validator(mode="after")
    def _documents_match_embeddings(self) -> Self:
        embedded = [item.image_id for item in self.embeddings]
        projected = [item.image_id for item in self.documents]
        if projected != embedded:
            raise ValueError("snapshot documents must match embedding image order")
        return self


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
    dense: list[LocalDense] = []


class LocalDense(_Frozen):
    retriever: Literal["bge"]
    version: str = Field(min_length=1)
    model: str
    dimension: int = Field(gt=0)
    encoder: Encoder
    document_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    projection_version: int = Field(ge=1)
    render_version: int = Field(ge=1)
    count: int = Field(ge=0)
    vectors_path: str
    mapping_path: str
    metadata_path: str


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
    base_image: str | None = None
    base_text: str | None = None
    base_rows: int = Field(default=0, ge=0)


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
    dense_counts: dict[Literal["bge"], int] = {}
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
    seq: int = Field(ge=0)
    image_key: str = Field(min_length=1)
    text_key: str = Field(min_length=1)
    base_image_key: str | None = Field(default=None, min_length=1)
    base_text_key: str | None = Field(default=None, min_length=1)
    base_rows: int = Field(default=0, ge=0)
    sealed: bool = False
    members: list[SealMember]

    @model_validator(mode="after")
    def _has_members(self) -> Self:
        if not self.members:
            raise ValueError("a shard needs at least one member")
        if (self.base_image_key is None) != (self.base_rows == 0):
            raise ValueError("a shard rewrite needs both a base object and its row count")
        if self.base_image_key is None and self.seq != 0:
            raise ValueError("only a rewrite advances the object generation")
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
    seq: int = Field(ge=0)
    rows: int = Field(ge=0)


class SealResult(_Frozen):
    shards: list[SealedShard] = []
    error: str | None = None
