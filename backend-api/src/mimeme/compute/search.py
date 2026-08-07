"""Blocking ONNX and FAISS state owned only by the resident search child."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol, cast

import faiss  # type: ignore[import-untyped]
import numpy as np
import onnxruntime as ort
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from mimeme.search.error import Failed, Incompatible, Loading, NotFound, Stale, Unavailable
from mimeme.search.model import (
    Batch,
    Candidate,
    CandidateRequest,
    ChildCall,
    Encoder,
    LoadCall,
    Loaded,
    PreparedLoad,
    QueryCall,
    RollbackCall,
    Status,
    StatusCall,
    SwitchCall,
)

_FAISS_VERSION = "1.13.2"
_ONNX_VERSION = "1.27.0"
_MAX_CURSOR_STATES = 256
_CHILD_CALL = TypeAdapter(ChildCall)


class _Metadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2]
    version: str
    embed_model: str
    dimension: int = Field(gt=0)
    metric: Literal["inner_product"]
    normalized: Literal[True]
    faiss_version: str
    onnxruntime_version: str
    encoder_repo: str
    encoder_revision: str
    encoder_variant: str
    kind: Literal["image", "text"] = "image"


class _TextEncoder(Protocol):
    source_model: str

    def encode(self, text: str) -> np.ndarray: ...


EncoderFactory = Callable[[Encoder], _TextEncoder]


class _Index:
    def __init__(self, index: faiss.Index, mapping: dict[int, int]) -> None:
        self.index = index
        self.mapping = mapping
        self.reverse = {image_id: row for row, image_id in mapping.items()}

    @classmethod
    def read(
        cls,
        index_path: Path,
        mapping_path: Path,
        *,
        dimension: int,
        hnsw_ef_search: int,
    ) -> _Index:
        index = faiss.read_index(str(index_path))
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise Incompatible("FAISS index metric must be inner product")
        if isinstance(index, faiss.IndexHNSW):
            index.hnsw.efSearch = hnsw_ef_search
        if index.d != dimension:
            raise Incompatible(
                f"index dimension {index.d} does not match metadata dimension {dimension}"
            )
        raw = json.loads(mapping_path.read_text(encoding="utf-8"))
        mapping = {int(row): int(image_id) for row, image_id in raw.items()}
        if index.ntotal != len(mapping) or set(mapping) != set(range(index.ntotal)):
            raise Incompatible("index mapping does not cover every FAISS row exactly once")
        image_ids = list(mapping.values())
        if any(image_id <= 0 for image_id in image_ids) or len(image_ids) != len(set(image_ids)):
            raise Incompatible("index mapping must contain unique positive stable image IDs")
        return cls(index, mapping)

    @property
    def size(self) -> int:
        return self.index.ntotal

    def search(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        query = _normalize(vector)
        distances, indices = self.index.search(query, min(k, self.index.ntotal))  # type: ignore[call-arg]
        return [
            (self.mapping[int(row)], float(score))
            for row, score in zip(indices[0], distances[0])
            if int(row) >= 0 and int(row) in self.mapping
        ]

    def vector(self, image_id: int) -> np.ndarray | None:
        row = self.reverse.get(image_id)
        if row is None:
            return None
        return cast(np.ndarray, self.index.reconstruct(row))  # type: ignore[call-arg]


class _Generation:
    def __init__(
        self,
        *,
        version: str,
        metadata: _Metadata,
        image: _Index,
        text: _Index | None,
        encoder: _TextEncoder,
        encoder_revision: str,
    ) -> None:
        self.version = version
        self.metadata = metadata
        self.image = image
        self.text = text
        self.encoder = encoder
        self.encoder_revision = encoder_revision


class _Cursor:
    def __init__(
        self,
        *,
        version: str,
        query: str,
        candidates: list[Candidate],
        offset: int,
        rank_k: int,
        max_rank: int,
    ) -> None:
        self.version = version
        self.query = query
        self.candidates = candidates
        self.offset = offset
        self.rank_k = rank_k
        self.max_rank = max_rank


class Resident:
    def __init__(self, *, encoder_factory: EncoderFactory | None = None) -> None:
        self._encoder_factory = encoder_factory or _load_encoder
        self._active: _Generation | None = None
        self._candidate: _Generation | None = None
        self._retained: _Generation | None = None
        self._cursors: dict[str, _Cursor] = {}

    def status(self) -> Status:
        active = self._active
        return Status(
            ready=active is not None,
            serving_version=active.version if active else None,
            candidate_version=self._candidate.version if self._candidate else None,
            retained_version=self._retained.version if self._retained else None,
            embed_model=active.metadata.embed_model if active else None,
            encoder_repo=active.metadata.encoder_repo if active else None,
            encoder_revision=active.encoder_revision if active else None,
            encoder_variant=active.metadata.encoder_variant if active else None,
            detail=None if active else "no index generation is serving",
        )

    def load(self, call: PreparedLoad) -> Loaded:
        generation = self._read_generation(call)
        self._candidate = generation
        return Loaded(
            version=generation.version,
            embed_model=generation.metadata.embed_model,
            dimension=generation.metadata.dimension,
            image_count=generation.image.size,
            text_count=generation.text.size if generation.text else None,
            faiss_version=generation.metadata.faiss_version,
            onnxruntime_version=generation.metadata.onnxruntime_version,
            encoder_revision=generation.encoder_revision,
        )

    def switch(self, version: str) -> Status:
        candidate = self._candidate
        if candidate is None:
            raise Loading("no candidate generation is loaded")
        if candidate.version != version:
            raise Stale(
                f"loaded candidate {candidate.version!r} does not match requested {version!r}"
            )
        self._retained = self._active
        self._active = candidate
        self._candidate = None
        self._cursors.clear()
        return self.status()

    def rollback(self, failed_version: str) -> Status:
        active = self._active
        if active is None or active.version != failed_version:
            raise Stale(f"generation {failed_version!r} is not currently serving")
        retained = self._retained
        if retained is None:
            raise Unavailable("no retained generation is available for rollback")
        self._candidate = active
        self._active = retained
        self._retained = None
        self._cursors.clear()
        return self.status()

    def query(self, request: CandidateRequest) -> Batch:
        active = self._active
        if active is None:
            raise Unavailable("search index not loaded")
        fingerprint = request.query.model_dump_json()
        if request.cursor is None:
            rank_k = min(request.count, self._max_rank(active, request))
            candidates = self._rank(active, request, rank_k)
            offset = 0
            max_rank = self._max_rank(active, request)
        else:
            state = self._cursors.pop(request.cursor, None)
            if state is None:
                raise Stale("search cursor is missing or has already been consumed")
            if state.version != active.version or state.query != fingerprint:
                raise Stale("search cursor does not match the active query")
            candidates = state.candidates
            offset = state.offset
            rank_k = state.rank_k
            max_rank = state.max_rank

        if offset + request.count > len(candidates) and rank_k < max_rank:
            rank_k = min(max_rank, rank_k + request.count)
            expanded = self._rank(active, request, rank_k)
            emitted_ids = {candidate.image_id for candidate in candidates[:offset]}
            candidates = candidates[:offset] + [
                candidate for candidate in expanded if candidate.image_id not in emitted_ids
            ]

        end = min(len(candidates), offset + request.count)
        exhausted = end >= len(candidates) and rank_k >= max_rank
        cursor: str | None = None
        if not exhausted:
            cursor = uuid.uuid4().hex
            if len(self._cursors) >= _MAX_CURSOR_STATES:
                self._cursors.pop(next(iter(self._cursors)))
            self._cursors[cursor] = _Cursor(
                version=active.version,
                query=fingerprint,
                candidates=candidates,
                offset=end,
                rank_k=rank_k,
                max_rank=max_rank,
            )
        return Batch(
            candidates=candidates[offset:end],
            cursor=cursor,
            exhausted=exhausted,
            version=active.version,
        )

    def _max_rank(self, generation: _Generation, request: CandidateRequest) -> int:
        if request.query.mode == "hybrid" and generation.text is not None:
            return max(generation.image.size, generation.text.size)
        return generation.image.size

    def _rank(self, generation: _Generation, request: CandidateRequest, k: int) -> list[Candidate]:
        query = request.query
        if query.similar_image_id is not None:
            vector = generation.image.vector(query.similar_image_id)
            if vector is None:
                raise NotFound(f"image {query.similar_image_id} is not in the active index")
            ranked = [
                pair
                for pair in generation.image.search(vector, k + 1)
                if pair[0] != query.similar_image_id
            ][:k]
        else:
            assert query.text is not None
            try:
                vector = generation.encoder.encode(query.text)
            except Exception as exc:
                raise Failed(f"query encoding failed: {exc}") from exc
            image = generation.image.search(vector, k)
            if query.mode == "hybrid" and generation.text is not None:
                ranked = _rrf(image, generation.text.search(vector, k))
            else:
                ranked = image
        return [Candidate(image_id=image_id, score=score) for image_id, score in ranked]

    def _read_generation(self, call: PreparedLoad) -> _Generation:
        paths = {name: Path(path) for name, path in call.paths.items()}
        required = {"index.faiss", "mapping.json", "metadata.json"}
        if not required.issubset(paths):
            raise Incompatible("image index generation is incomplete")
        try:
            metadata = _Metadata.model_validate_json(
                paths["metadata.json"].read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise Incompatible(f"invalid index metadata: {exc}") from exc
        _validate_metadata(metadata, call.version, kind="image")
        image = _Index.read(
            paths["index.faiss"],
            paths["mapping.json"],
            dimension=metadata.dimension,
            hnsw_ef_search=call.hnsw_ef_search,
        )

        text: _Index | None = None
        text_names = {"text_index.faiss", "text_mapping.json", "text_metadata.json"}
        present = text_names.intersection(paths)
        if present and present != text_names:
            raise Incompatible("text index generation is incomplete")
        if present:
            try:
                text_metadata = _Metadata.model_validate_json(
                    paths["text_metadata.json"].read_text(encoding="utf-8")
                )
            except Exception as exc:
                raise Incompatible(f"invalid text index metadata: {exc}") from exc
            _validate_metadata(text_metadata, call.version, kind="text")
            if text_metadata.embed_model != metadata.embed_model:
                raise Incompatible("image and text indexes use different embedding models")
            if text_metadata.dimension != metadata.dimension:
                raise Incompatible("image and text indexes use different dimensions")
            text = _Index.read(
                paths["text_index.faiss"],
                paths["text_mapping.json"],
                dimension=text_metadata.dimension,
                hnsw_ef_search=call.hnsw_ef_search,
            )

        try:
            encoder = self._encoder_factory(call.encoder)
        except Exception as exc:
            raise Failed(f"text encoder load failed: {exc}") from exc
        if encoder.source_model != metadata.embed_model:
            raise Incompatible(
                f"encoder model {encoder.source_model!r} does not match index model "
                f"{metadata.embed_model!r}"
            )
        if (
            metadata.encoder_repo != call.encoder.repo
            or metadata.encoder_revision != call.encoder.revision
            or metadata.encoder_variant != call.encoder.variant
        ):
            raise Incompatible("encoder artifact identity does not match index metadata")
        try:
            warmup = _normalize(encoder.encode("warmup"))
        except Exception as exc:
            raise Failed(f"text encoder warmup failed: {exc}") from exc
        if warmup.shape[1] != metadata.dimension:
            raise Incompatible(
                f"encoder dimension {warmup.shape[1]} does not match index dimension "
                f"{metadata.dimension}"
            )
        return _Generation(
            version=call.version,
            metadata=metadata,
            image=image,
            text=text,
            encoder=encoder,
            encoder_revision=call.encoder.revision,
        )


def _validate_metadata(metadata: _Metadata, version: str, *, kind: str) -> None:
    if metadata.version != version:
        raise Incompatible(
            f"metadata version {metadata.version!r} does not match generation {version!r}"
        )
    if metadata.kind != kind:
        raise Incompatible(f"expected {kind} metadata, got {metadata.kind}")
    if metadata.faiss_version != _FAISS_VERSION:
        raise Incompatible(
            f"index requires FAISS {metadata.faiss_version}, runtime is {_FAISS_VERSION}"
        )
    if metadata.onnxruntime_version != _ONNX_VERSION:
        raise Incompatible(
            "index requires ONNX Runtime "
            f"{metadata.onnxruntime_version}, runtime is {_ONNX_VERSION}"
        )


def _normalize(vector: np.ndarray) -> np.ndarray:
    query = np.asarray(vector, dtype=np.float32)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.ndim != 2 or query.shape[0] != 1:
        raise Incompatible("query encoder must return one vector")
    norm = np.linalg.norm(query, axis=1, keepdims=True)
    if not np.all(np.isfinite(query)) or np.any(norm == 0):
        raise Incompatible("query encoder returned an invalid vector")
    return query / norm


def _rrf(
    image: list[tuple[int, float]], text: list[tuple[int, float]], *, k: int = 60
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rank, (image_id, _) in enumerate(image, start=1):
        scores[image_id] = scores.get(image_id, 0.0) + 1 / (rank + k)
    for rank, (image_id, _) in enumerate(text, start=1):
        scores[image_id] = scores.get(image_id, 0.0) + 1 / (rank + k)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class _OnnxEncoder:
    def __init__(self, config: Encoder) -> None:
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        root = Path(
            snapshot_download(
                config.repo,
                revision=config.revision,
                allow_patterns=[config.variant, "tokenizer.json", "export_meta.json"],
            )
        )
        meta = json.loads((root / "export_meta.json").read_text(encoding="utf-8"))
        self.source_model = str(meta["source_model"])
        self._tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self._tokenizer.enable_padding(
            length=int(meta.get("max_length", 64)),
            pad_id=int(meta.get("pad_token_id", 0)),
            pad_token="<pad>",
        )
        self._tokenizer.enable_truncation(max_length=int(meta.get("max_length", 64)))
        options = ort.SessionOptions()
        options.intra_op_num_threads = config.threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            str(root / config.variant),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def encode(self, text: str) -> np.ndarray:
        tokens = np.array([self._tokenizer.encode(text).ids], dtype=np.int64)
        value = self._session.run(["text_embeds"], {"input_ids": tokens})[0]
        return cast(np.ndarray, value)[0].astype(np.float32)


def _load_encoder(config: Encoder) -> _TextEncoder:
    return _OnnxEncoder(config)


def dispatch(resident: Resident, raw: bytes) -> dict:
    call = _CHILD_CALL.validate_json(raw)
    if isinstance(call, StatusCall):
        return resident.status().model_dump()
    if isinstance(call, LoadCall):
        return resident.load(call.load).model_dump()
    if isinstance(call, SwitchCall):
        return resident.switch(call.version).model_dump()
    if isinstance(call, RollbackCall):
        return resident.rollback(call.failed_version).model_dump()
    if isinstance(call, QueryCall):
        return resident.query(call.request).model_dump()
    raise Failed(f"unknown search operation: {call!r}")
