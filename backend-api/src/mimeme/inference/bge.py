from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mimeme.search.document import SearchDocument

SOURCE_MODEL = "BAAI/bge-small-en-v1.5"
SOURCE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DIMENSION = 384
MAX_LENGTH = 256
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RENDER_VERSION = 1
OWNED_REPO = "listlessbird/bge-small-en-v1.5-onnx"
OWNED_REVISION = "d46fcc3e67304e574e08e911ce7e50d71bb728cf"
MODEL_SHA256 = "6fb40fbcdf3dcc7a3fed12d56ff2d1324f69d0b7fd6c5afe05f4530a6142fdf8"
TOKENIZER_SHA256 = "0d3aef594edd5f9b53e7f814277a9171dc70ff93eb66bda6e01f7aa53997d963"
EXPORT_META_SHA256 = "619bafc6963e85a17c13e58b30cd8ac11ad0b5d737d70395bda9e06655883858"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Export(_Frozen):
    source_model: Literal["BAAI/bge-small-en-v1.5"] = SOURCE_MODEL
    source_revision: Literal["5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"] = SOURCE_REVISION
    repo: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    variant: str = Field(min_length=1)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    export_meta_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    opset: int = Field(ge=1)
    dimension: Literal[384] = DIMENSION
    max_length: Literal[256] = MAX_LENGTH
    query_prefix: Literal["Represent this sentence for searching relevant passages: "] = (
        QUERY_PREFIX
    )
    pooling: Literal["cls"] = "cls"
    normalization: Literal["l2"] = "l2"
    quantization: Literal["int8-dynamic"] = "int8-dynamic"


EXPORT = Export(
    repo=OWNED_REPO,
    revision=OWNED_REVISION,
    variant="model-int8.onnx",
    model_sha256=MODEL_SHA256,
    tokenizer_sha256=TOKENIZER_SHA256,
    export_meta_sha256=EXPORT_META_SHA256,
    opset=18,
)


class CorpusItem(_Frozen):
    image_id: int = Field(gt=0)
    text: str


class EncodeBatch(_Frozen):
    document_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    projection_version: Literal[1] = 1
    render_version: Literal[1] = RENDER_VERSION
    export: Export
    items: tuple[CorpusItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_images(self) -> Self:
        image_ids = [item.image_id for item in self.items]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("BGE batch image IDs must be unique")
        return self


class Vector(_Frozen):
    image_id: int = Field(gt=0)
    values: tuple[float, ...] = Field(min_length=DIMENSION, max_length=DIMENSION)

    @model_validator(mode="after")
    def _normalized(self) -> Self:
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("BGE vector values must be finite")
        norm = math.sqrt(sum(value * value for value in self.values))
        if not 0.999 <= norm <= 1.001:
            raise ValueError("BGE vectors must be L2-normalized")
        return self


class EncodedBatch(_Frozen):
    document_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    export: Export
    items: tuple[Vector, ...]


def validate_result(request: EncodeBatch, result: EncodedBatch) -> None:
    if result.document_content_sha256 != request.document_content_sha256:
        raise ValueError("BGE result document checksum does not match its request")
    if result.export != request.export:
        raise ValueError("BGE result export identity does not match its request")
    requested = [item.image_id for item in request.items]
    returned = [item.image_id for item in result.items]
    if returned != requested or len(returned) != len(set(returned)):
        raise ValueError("BGE result image IDs do not exactly match the requested order")


def render_document(value: SearchDocument) -> str:
    fields = (
        ("Titles", value.titles),
        ("Tags", value.tags),
        ("Captions", value.captions),
        ("OCR", value.ocr_texts),
        ("Categories", value.categories),
        ("Types", value.types),
        ("Origins", value.origins),
        ("Years", value.years),
        ("Descriptions", value.descriptions),
    )
    return "\n".join(f"{label}: {'; '.join(values)}" for label, values in fields if values)


def render_query(text: str) -> str:
    return f"{QUERY_PREFIX}{text}"


def pool(last_hidden_state: object) -> object:
    import numpy as np

    hidden = np.asarray(last_hidden_state, dtype=np.float32)
    if hidden.ndim != 3 or hidden.shape[0] == 0 or hidden.shape[1] == 0:
        raise ValueError("BGE output must be a non-empty batch of token embeddings")
    if hidden.shape[2] != DIMENSION:
        raise ValueError(f"BGE output dimension must be {DIMENSION}")
    vectors = hidden[:, 0, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.all(np.isfinite(vectors)) or np.any(norms == 0):
        raise ValueError("BGE output contains an invalid CLS embedding")
    return vectors / norms
