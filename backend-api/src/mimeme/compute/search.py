from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol, cast

import faiss  # type: ignore[import-untyped]
import numpy as np
import onnxruntime as ort
import structlog
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from mimeme.compute import encoder_cache
from mimeme.index import bm25
from mimeme.inference import bge
from mimeme.search import fusion, generation_workspace, recipe, retriever
from mimeme.search.error import Failed, Incompatible, Loading, NotFound, Stale, Unavailable
from mimeme.search.model import (
    Batch,
    Candidate,
    CandidateRequest,
    ChildCall,
    ClearCall,
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
    kind: Literal["image", "bge"] = "image"


class _TextEncoder(encoder_cache.Session, Protocol): ...


EncoderFactory = Callable[[Encoder], encoder_cache.Session]


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
        lexical: bm25.Index | None,
        encoder: encoder_cache.Lease,
        dense: dict[recipe.RetrieverId, _Index],
        dense_encoders: dict[recipe.RetrieverId, encoder_cache.Lease],
        workspace: generation_workspace.Workspace,
    ) -> None:
        self.version = version
        self.metadata = metadata
        self.image = image
        self.lexical = lexical
        self.encoder = encoder
        self.dense = dense
        self.dense_encoders = dense_encoders
        self.workspace = workspace


class _Cursor:
    def __init__(
        self,
        *,
        version: str,
        query: str,
        candidates: list[Candidate],
        offset: int,
        definition: recipe.Definition,
    ) -> None:
        self.version = version
        self.query = query
        self.candidates = candidates
        self.offset = offset
        self.definition = definition


class _VectorRetriever:
    def __init__(self, retriever_id: recipe.RetrieverId, index: _Index) -> None:
        self.id = retriever_id
        self._index = index

    def search(self, context: np.ndarray, *, depth: int) -> list[retriever.Scored]:
        return [
            retriever.Scored(image_id=image_id, score=score)
            for image_id, score in self._index.search(context, depth)
        ]


class Resident:
    def __init__(self, *, encoder_factory: EncoderFactory | None = None) -> None:
        self._encoders = encoder_cache.create(encoder_factory or _load_encoder)
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
            encoder_revision=(encoder_cache.identity(active.encoder).revision if active else None),
            encoder_variant=active.metadata.encoder_variant if active else None,
            bm25_available=active.lexical is not None if active else False,
            bge_available="bge" in active.dense if active else False,
            detail=None if active else "no index generation is serving",
        )

    def load(self, call: PreparedLoad) -> Loaded:
        started = time.perf_counter()
        identity = encoder_cache.identify(call.encoder)
        try:
            generation = self._read_generation(call)
            previous = self._candidate
            self._candidate = generation
            if previous is not None:
                self._evict(previous, reason="candidate_replaced")
            result = Loaded(
                version=generation.version,
                embed_model=generation.metadata.embed_model,
                dimension=generation.metadata.dimension,
                image_count=generation.image.size,
                bm25_count=call.bm25.count if call.bm25 is not None else None,
                dense_counts={
                    item.retriever: generation.dense[item.retriever].size for item in call.dense
                },
                faiss_version=generation.metadata.faiss_version,
                onnxruntime_version=generation.metadata.onnxruntime_version,
                encoder_revision=identity.revision,
            )
        except Exception as exc:
            _emit_resource_event(
                "search_generation_load",
                started=started,
                version=call.version,
                identity=identity,
                outcome="failed",
                error_type=type(exc).__name__,
            )
            raise
        _emit_resource_event(
            "search_generation_load",
            started=started,
            version=call.version,
            identity=identity,
            outcome="loaded",
        )
        return result

    def switch(self, version: str) -> Status:
        started = time.perf_counter()
        candidate = self._candidate
        if candidate is None:
            raise Loading("no candidate generation is loaded")
        if candidate.version != version:
            raise Stale(
                f"loaded candidate {candidate.version!r} does not match requested {version!r}"
            )
        previous_retained = self._retained
        self._retained = self._active
        self._active = candidate
        self._candidate = None
        self._cursors.clear()
        if previous_retained is not None:
            self._evict(previous_retained, reason="retained_replaced")
        status = self.status()
        _emit_resource_event(
            "search_generation_switch",
            started=started,
            version=candidate.version,
            identity=encoder_cache.identity(candidate.encoder),
            outcome="serving",
        )
        return status

    def rollback(self, failed_version: str) -> Status:
        started = time.perf_counter()
        active = self._active
        if active is None or active.version != failed_version:
            raise Stale(f"generation {failed_version!r} is not currently serving")
        retained = self._retained
        if retained is None:
            raise Unavailable("no retained generation is available for rollback")
        previous_candidate = self._candidate
        self._candidate = active
        self._active = retained
        self._retained = None
        self._cursors.clear()
        if previous_candidate is not None:
            self._evict(previous_candidate, reason="rollback_candidate_replaced")
        status = self.status()
        _emit_resource_event(
            "search_generation_rollback",
            started=started,
            version=retained.version,
            identity=encoder_cache.identity(retained.encoder),
            outcome="serving",
        )
        return status

    def clear(self) -> Status:
        generations = (self._candidate, self._active, self._retained)
        self._candidate = None
        self._active = None
        self._retained = None
        self._cursors.clear()
        seen: set[int] = set()
        for generation in generations:
            if generation is not None and id(generation) not in seen:
                seen.add(id(generation))
                self._evict(generation, reason="clear")
        return self.status()

    def query(self, request: CandidateRequest) -> Batch:
        active = self._active
        if active is None:
            raise Unavailable("search index not loaded")
        fingerprint = request.query.model_dump_json()
        if request.cursor is None:
            available: set[recipe.RetrieverId] = {"siglip_image"}
            if active.lexical is not None:
                available.add("bm25")
            available.update(active.dense)
            definition = recipe.resolve(request.query.recipe_id, available=available)
            candidates = self._rank(active, request, definition)
            offset = 0
        else:
            state = self._cursors.pop(request.cursor, None)
            if state is None:
                raise Stale("search cursor is missing or has already been consumed")
            if state.version != active.version or state.query != fingerprint:
                raise Stale("search cursor does not match the active query")
            candidates = state.candidates
            offset = state.offset
            definition = state.definition

        end = min(len(candidates), offset + request.count)
        exhausted = end >= len(candidates)
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
                definition=definition,
            )
        return Batch(
            candidates=candidates[offset:end],
            cursor=cursor,
            exhausted=exhausted,
            version=active.version,
        )

    def _rank(
        self,
        generation: _Generation,
        request: CandidateRequest,
        definition: recipe.Definition,
    ) -> list[Candidate]:
        query = request.query
        if query.similar_image_id is not None:
            vector = generation.image.vector(query.similar_image_id)
            if vector is None:
                raise NotFound(f"image {query.similar_image_id} is not in the active index")
            ranked = [
                pair
                for pair in generation.image.search(vector, definition.candidate_depth + 1)
                if pair[0] != query.similar_image_id
            ][: definition.candidate_depth]
            return [Candidate(image_id=image_id, score=score) for image_id, score in ranked]
        else:
            assert query.text is not None
            indexes = {
                "siglip_image": generation.image,
                **generation.dense,
            }
            vectors: dict[str, np.ndarray] = {}
            rankings: list[list[retriever.Scored]] = []
            for retriever_id in definition.retrievers:
                if retriever_id == "bm25":
                    assert generation.lexical is not None
                    rankings.append(
                        [
                            retriever.Scored(image_id=image_id, score=-float(rank))
                            for rank, image_id in enumerate(
                                bm25.query(
                                    generation.lexical,
                                    query.text,
                                    depth=definition.candidate_depth,
                                ),
                                start=1,
                            )
                        ]
                    )
                    continue
                index = indexes[retriever_id]
                assert index is not None
                encoder = (
                    generation.dense_encoders[retriever_id]
                    if retriever_id in generation.dense_encoders
                    else generation.encoder
                )
                identity = encoder_cache.identity(encoder)
                cache_key = f"{identity.repo}@{identity.revision}/{identity.variant}"
                try:
                    vector = vectors.get(cache_key)
                    if vector is None:
                        vector = encoder_cache.encode(encoder, query.text)
                        vectors[cache_key] = vector
                except Exception as exc:
                    raise Failed(f"{retriever_id} query encoding failed: {exc}") from exc
                rankings.append(
                    _VectorRetriever(retriever_id, index).search(
                        vector,
                        depth=definition.candidate_depth,
                    )
                )
            if len(rankings) == 1:
                return [Candidate(image_id=item.image_id, score=item.score) for item in rankings[0]]
            fused = fusion.rrf(
                [[item.image_id for item in ranking] for ranking in rankings],
                k=definition.rrf_k,
            )
            return [Candidate(image_id=item.image_id, score=item.score) for item in fused]

    def _read_generation(self, call: PreparedLoad) -> _Generation:
        paths = {name: Path(path) for name, path in call.paths.items()}
        try:
            workspace = generation_workspace.claim(Path(call.workspace), paths.values())
        except Exception as exc:
            raise Incompatible(f"invalid generation workspace: {exc}") from exc
        try:
            return self._read_claimed_generation(call, paths, workspace)
        except Exception:
            generation_workspace.release(workspace)
            raise

    def _read_claimed_generation(
        self,
        call: PreparedLoad,
        paths: dict[str, Path],
        workspace: generation_workspace.Workspace,
    ) -> _Generation:
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

        dense: dict[recipe.RetrieverId, _Index] = {}
        dense_metadata: dict[recipe.RetrieverId, _Metadata] = {}
        for descriptor in call.dense:
            names = {file.name for file in descriptor.files}
            if not names.issubset(paths):
                raise Incompatible(f"{descriptor.retriever} index generation is incomplete")
            metadata_name = f"{descriptor.retriever}_metadata.json"
            try:
                item_metadata = _Metadata.model_validate_json(
                    paths[metadata_name].read_text(encoding="utf-8")
                )
            except Exception as exc:
                raise Incompatible(f"invalid {descriptor.retriever} index metadata: {exc}") from exc
            _validate_metadata(item_metadata, call.version, kind=descriptor.retriever)
            if (
                item_metadata.embed_model != descriptor.model
                or item_metadata.dimension != descriptor.dimension
                or item_metadata.encoder_repo != descriptor.encoder.repo
                or item_metadata.encoder_revision != descriptor.encoder.revision
                or item_metadata.encoder_variant != descriptor.encoder.variant
            ):
                raise Incompatible(
                    f"{descriptor.retriever} index metadata does not match its descriptor"
                )
            dense[descriptor.retriever] = _Index.read(
                paths[f"{descriptor.retriever}_index.faiss"],
                paths[f"{descriptor.retriever}_mapping.json"],
                dimension=descriptor.dimension,
                hnsw_ef_search=call.hnsw_ef_search,
            )
            if dense[descriptor.retriever].size != descriptor.count:
                raise Incompatible(f"{descriptor.retriever} index count is incompatible")
            dense_metadata[descriptor.retriever] = item_metadata

        lexical = None
        if call.bm25 is not None:
            path = paths.get(call.bm25.name)
            if path is None:
                raise Incompatible("BM25 generation artifact is missing")
            try:
                lexical = bm25.open(
                    path,
                    count=call.bm25.count,
                    sqlite_version=call.bm25.sqlite_version,
                )
            except Exception as exc:
                raise Incompatible(f"BM25 generation is invalid: {exc}") from exc

        leases: list[encoder_cache.Lease] = []
        try:
            encoder = encoder_cache.acquire(self._encoders, call.encoder)
            leases.append(encoder)
        except Exception as exc:
            if lexical is not None:
                bm25.close(lexical)
            raise Failed(f"text encoder load failed: {exc}") from exc
        try:
            _validate_encoder(encoder, call.encoder, metadata, label="text")
            dense_encoders: dict[recipe.RetrieverId, encoder_cache.Lease] = {}
            for descriptor in call.dense:
                lease = encoder_cache.acquire(self._encoders, descriptor.encoder)
                leases.append(lease)
                _validate_encoder(
                    lease,
                    descriptor.encoder,
                    dense_metadata[descriptor.retriever],
                    label=descriptor.retriever,
                )
                dense_encoders[descriptor.retriever] = lease
        except Exception:
            if lexical is not None:
                bm25.close(lexical)
            for lease in leases:
                encoder_cache.release(lease)
            raise
        return _Generation(
            version=call.version,
            metadata=metadata,
            image=image,
            lexical=lexical,
            encoder=encoder,
            dense=dense,
            dense_encoders=dense_encoders,
            workspace=workspace,
        )

    def _evict(self, generation: _Generation, *, reason: str) -> None:
        started = time.perf_counter()
        identity = encoder_cache.identity(generation.encoder)
        if generation.lexical is not None:
            bm25.close(generation.lexical)
        encoder_cache.release(generation.encoder)
        for encoder in generation.dense_encoders.values():
            encoder_cache.release(encoder)
        generation_workspace.release(generation.workspace)
        _emit_resource_event(
            "search_generation_evicted",
            started=started,
            version=generation.version,
            identity=identity,
            outcome="evicted",
            reason=reason,
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


def _validate_encoder(
    lease: encoder_cache.Lease,
    config: Encoder,
    metadata: _Metadata,
    *,
    label: str,
) -> None:
    source_model = encoder_cache.source_model(lease)
    if source_model != metadata.embed_model:
        raise Incompatible(
            f"{label} encoder model {source_model!r} does not match index model "
            f"{metadata.embed_model!r}"
        )
    if (
        metadata.encoder_repo != config.repo
        or metadata.encoder_revision != config.revision
        or metadata.encoder_variant != config.variant
    ):
        raise Incompatible(f"{label} encoder artifact identity does not match index metadata")
    try:
        warmup = _normalize(encoder_cache.encode(lease, "warmup"))
    except Exception as exc:
        raise Failed(f"{label} encoder warmup failed: {exc}") from exc
    if warmup.shape[1] != metadata.dimension:
        raise Incompatible(
            f"{label} encoder dimension {warmup.shape[1]} does not match index dimension "
            f"{metadata.dimension}"
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
        if config.repo == bge.EXPORT.repo and config.revision == bge.EXPORT.revision:
            _validate_bge_export(root, config, meta)
        self.source_model = str(meta["source_model"])
        self._query_prefix = str(meta.get("query_prefix", ""))
        self._input_names = tuple(meta.get("input_names", ["input_ids"]))
        self._output_name = str(meta.get("output_name", "text_embeds"))
        self._tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        if self._input_names == ("input_ids",):
            self._tokenizer.enable_padding(
                length=int(meta.get("max_length", 64)),
                pad_id=int(meta.get("pad_token_id", 0)),
                pad_token=str(meta.get("pad_token", "[PAD]")),
            )
        else:
            self._tokenizer.no_padding()
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
        encoded = self._tokenizer.encode(f"{self._query_prefix}{text}")
        fields = {
            "input_ids": encoded.ids,
            "attention_mask": encoded.attention_mask,
            "token_type_ids": encoded.type_ids,
        }
        inputs = {name: np.asarray([fields[name]], dtype=np.int64) for name in self._input_names}
        value = self._session.run([self._output_name], inputs)[0]
        return cast(np.ndarray, value)[0].astype(np.float32)


def _load_encoder(config: Encoder) -> _TextEncoder:
    return _OnnxEncoder(config)


def _validate_bge_export(root: Path, config: Encoder, meta: dict[str, object]) -> None:
    expected_hashes = {
        config.variant: bge.MODEL_SHA256,
        "tokenizer.json": bge.TOKENIZER_SHA256,
        "export_meta.json": bge.EXPORT_META_SHA256,
    }
    for name, expected in expected_hashes.items():
        if _digest(root / name) != expected:
            raise Incompatible(f"BGE encoder artifact checksum mismatch: {name}")
    expected_meta: dict[str, object] = {
        "source_model": bge.SOURCE_MODEL,
        "source_revision": bge.SOURCE_REVISION,
        "opset": bge.EXPORT.opset,
        "tokenizer_sha256": bge.TOKENIZER_SHA256,
        "max_length": bge.MAX_LENGTH,
        "query_prefix": bge.QUERY_PREFIX,
        "pooling": "cls",
        "normalization": "l2",
        "dimension": bge.DIMENSION,
        "quantization": "int8-dynamic",
        "input_names": ["input_ids", "attention_mask", "token_type_ids"],
        "output_name": "embeddings",
    }
    if any(meta.get(name) != value for name, value in expected_meta.items()):
        raise Incompatible("BGE encoder export metadata is incompatible")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rss_bytes() -> int:
    fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
    return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")


def _emit_resource_event(
    event: str,
    *,
    started: float,
    version: str,
    identity: encoder_cache.Identity,
    outcome: str,
    **fields: object,
) -> None:
    structlog.get_logger().info(
        event,
        generation_version=version,
        encoder_repo=identity.repo,
        encoder_revision=identity.revision,
        encoder_variant=identity.variant,
        encoder_threads=identity.threads,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
        rss_bytes=_rss_bytes(),
        outcome=outcome,
        **fields,
    )


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
    if isinstance(call, ClearCall):
        return resident.clear().model_dump()
    if isinstance(call, QueryCall):
        return resident.query(call.request).model_dump()
    raise Failed(f"unknown search operation: {call!r}")
