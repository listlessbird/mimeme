from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import faiss  # type: ignore[import-untyped]
import numpy as np
import onnxruntime as ort

from mimeme.index.model import (
    Built,
    BuiltFile,
    Encoder,
    LocalDense,
    LocalEmbedding,
    LocalMember,
    LocalShard,
    PackCall,
    Packed,
    PackedFile,
    PreparedBuild,
)


def build(request: PreparedBuild) -> Built:
    faiss.omp_set_num_threads(request.native_threads)
    root = Path(request.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    image_vectors, image_ids = _load(request, text=False)
    text_vectors, text_ids = _load(request, text=True)

    if request.embeddings and image_vectors.shape[1] != request.dimension:
        raise ValueError(
            f"image embedding dimension {image_vectors.shape[1]} does not match {request.dimension}"
        )

    files: list[Path] = []
    files.extend(
        _write_kind(
            root,
            request=request,
            vectors=image_vectors,
            image_ids=image_ids,
            kind="image",
        )
    )
    if len(text_ids) > 0:
        if text_vectors.shape[1] != request.dimension:
            raise ValueError("text and image embedding dimensions differ")
        files.extend(
            _write_kind(
                root,
                request=request,
                vectors=text_vectors,
                image_ids=text_ids,
                kind="text",
            )
        )

    dense_counts: dict[Literal["bge"], int] = {}
    for dense in request.dense:
        vectors, image_ids = _load_dense(dense)
        files.extend(
            _write_kind(
                root,
                request=request,
                vectors=vectors,
                image_ids=image_ids,
                kind=dense.retriever,
                model=dense.model,
                dimension=dense.dimension,
                encoder=dense.encoder,
            )
        )
        dense_counts[dense.retriever] = len(image_ids)

    return Built(
        version=request.version,
        target_generation=request.target_generation,
        model=request.model,
        index_type=request.index_type,
        dimension=request.dimension,
        image_count=len(image_ids),
        text_count=len(text_ids) or None,
        dense_counts=dense_counts,
        files=[_describe(path) for path in files],
    )


def pack(request: PackCall) -> Packed:
    if not request.members:
        raise ValueError("a shard needs at least one member")
    dimension = _dimension(request.members[0])
    base_images = _base(request.base_image, request.base_rows, dimension)
    base_texts = _base(request.base_text, request.base_rows, dimension)
    total = request.base_rows + len(request.members)
    images = np.empty((total, dimension), dtype=np.float32)
    texts = np.zeros((total, dimension), dtype=np.float32)
    if base_images is not None and base_texts is not None:
        images[: request.base_rows] = base_images
        texts[: request.base_rows] = base_texts
    for offset, member in enumerate(request.members):
        row = request.base_rows + offset
        images[row] = _read(member.image_path, member.image_id, dimension)
        if member.text_path is not None:
            texts[row] = _read(member.text_path, member.image_id, dimension)
    image_out = Path(request.image_out)
    text_out = Path(request.text_out)
    _save(image_out, images)
    _save(text_out, texts)
    return Packed(
        rows=total,
        dimension=dimension,
        image=_packed_file(image_out),
        text=_packed_file(text_out),
    )


def _base(path: str | None, rows: int, dimension: int) -> np.ndarray | None:
    if path is None:
        if rows:
            raise ValueError("a shard rewrite needs its base object")
        return None
    matrix = np.load(path, allow_pickle=False)
    if matrix.ndim != 2 or matrix.dtype != np.float32:
        raise ValueError("shard base is not a 2-D float32 matrix")
    if matrix.shape[0] != rows:
        raise ValueError(f"shard base has {matrix.shape[0]} rows, expected {rows}")
    if matrix.shape[1] != dimension:
        raise ValueError(f"shard base has dimension {matrix.shape[1]}, expected {dimension}")
    return matrix


def _dimension(member: LocalMember) -> int:
    vector = np.load(member.image_path, allow_pickle=False)
    if vector.ndim != 1 or vector.shape[0] == 0:
        raise ValueError(f"invalid embedding for image {member.image_id}")
    return int(vector.shape[0])


def _read(path: str, image_id: int, dimension: int) -> np.ndarray:
    vector = np.load(path, allow_pickle=False)
    if vector.ndim != 1 or not np.issubdtype(vector.dtype, np.floating):
        raise ValueError(f"embedding for image {image_id} is not a 1-D float vector")
    if vector.shape[0] != dimension:
        raise ValueError(
            f"embedding for image {image_id} has dimension {vector.shape[0]}, expected {dimension}"
        )
    return vector.astype(np.float32, copy=False)


def _save(path: Path, matrix: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
        handle.flush()


def _packed_file(path: Path) -> PackedFile:
    return PackedFile(path=str(path), sha256=_digest(path), length=path.stat().st_size)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(request: PreparedBuild, *, text: bool) -> tuple[np.ndarray, list[int]]:
    shards = {shard.number: shard for shard in request.shards}
    selected = [item for item in request.embeddings if _available(item, text=text)]
    if not selected:
        return np.empty((0, request.dimension), dtype=np.float32), []
    matrix = np.empty((len(selected), request.dimension), dtype=np.float32)
    opened: dict[str, np.ndarray] = {}
    for position, item in enumerate(selected):
        vector = _vector(item, shards, opened, text=text)
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            raise ValueError(f"invalid embedding for image {item.image_id}")
        if vector.shape[0] != request.dimension:
            raise ValueError(
                f"embedding for image {item.image_id} has dimension {vector.shape[0]}, "
                f"expected {request.dimension}"
            )
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError(f"zero embedding for image {item.image_id}")
        matrix[position] = vector / norm
    return matrix, [item.image_id for item in selected]


def _available(item: LocalEmbedding, *, text: bool) -> bool:
    if item.shard is not None:
        return item.text_present if text else True
    return item.text_path is not None if text else item.image_path is not None


def _vector(
    item: LocalEmbedding,
    shards: dict[int, LocalShard],
    opened: dict[str, np.ndarray],
    *,
    text: bool,
) -> np.ndarray:
    if item.shard is None:
        path = item.text_path if text else item.image_path
        assert path is not None
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    shard = shards.get(item.shard)
    if shard is None:
        raise ValueError(f"build is missing shard {item.shard} for image {item.image_id}")
    path = shard.text_path if text else shard.image_path
    if path is None:
        raise ValueError(f"build is missing the text half of shard {item.shard}")
    matrix = opened.get(path)
    if matrix is None:
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        if matrix.ndim != 2 or matrix.dtype != np.float32:
            raise ValueError(f"shard {item.shard} is not a 2-D float32 matrix")
        opened[path] = matrix
    assert item.row is not None
    if item.row >= matrix.shape[0]:
        raise ValueError(f"shard {item.shard} has no row {item.row}")
    return matrix[item.row]


def _write_kind(
    root: Path,
    *,
    request: PreparedBuild,
    vectors: np.ndarray,
    image_ids: list[int],
    kind: Literal["image", "text", "bge"],
    model: str | None = None,
    dimension: int | None = None,
    encoder: Encoder | None = None,
) -> list[Path]:
    resolved_dimension = dimension or request.dimension
    resolved_encoder = encoder or request.encoder
    index = _new_index(request.index_type, resolved_dimension)
    if len(image_ids):
        index.add(vectors)  # type: ignore[call-arg]
    prefix = "" if kind == "image" else f"{kind}_"
    index_path = root / f"{prefix}index.faiss"
    mapping_path = root / f"{prefix}mapping.json"
    metadata_path = root / f"{prefix}metadata.json"
    faiss.write_index(index, str(index_path))
    mapping_path.write_text(
        json.dumps({str(row): image_id for row, image_id in enumerate(image_ids)}, sort_keys=True),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "version": request.version,
                "embed_model": model or request.model,
                "dimension": resolved_dimension,
                "metric": "inner_product",
                "normalized": True,
                "faiss_version": faiss.__version__,
                "onnxruntime_version": ort.__version__,
                "encoder_repo": resolved_encoder.repo,
                "encoder_revision": resolved_encoder.revision,
                "encoder_variant": resolved_encoder.variant,
                "kind": kind,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return [index_path, mapping_path, metadata_path]


def _load_dense(value: LocalDense) -> tuple[np.ndarray, list[int]]:
    try:
        metadata = json.loads(Path(value.metadata_path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid {value.retriever} vector metadata: {exc}") from exc
    expected = {
        "schema_version": 1,
        "retriever": value.retriever,
        "version": value.version,
        "model": value.model,
        "dimension": value.dimension,
        "dtype": "float32",
        "normalized": True,
        "count": value.count,
        "document_content_sha256": value.document_content_sha256,
        "projection_version": value.projection_version,
        "render_version": value.render_version,
    }
    if any(metadata.get(name) != expected_value for name, expected_value in expected.items()):
        raise ValueError(f"{value.retriever} vector metadata does not match its descriptor")
    export = metadata.get("export")
    if not isinstance(export, dict) or (
        export.get("source_model") != value.model
        or export.get("repo") != value.encoder.repo
        or export.get("revision") != value.encoder.revision
        or export.get("variant") != value.encoder.variant
    ):
        raise ValueError(f"{value.retriever} vector export identity is incompatible")
    matrix = np.load(value.vectors_path, allow_pickle=False)
    if matrix.dtype != np.float32 or matrix.shape != (value.count, value.dimension):
        raise ValueError(f"{value.retriever} vector matrix has an incompatible shape or dtype")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{value.retriever} vector matrix contains non-finite values")
    if value.count and not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-3):
        raise ValueError(f"{value.retriever} vector matrix is not normalized")
    try:
        raw_mapping = json.loads(Path(value.mapping_path).read_text(encoding="utf-8"))
        mapping = {int(row): int(image_id) for row, image_id in raw_mapping.items()}
    except Exception as exc:
        raise ValueError(f"invalid {value.retriever} vector mapping: {exc}") from exc
    if set(mapping) != set(range(value.count)):
        raise ValueError(f"{value.retriever} vector mapping does not cover every row")
    image_ids = [mapping[row] for row in range(value.count)]
    if any(image_id <= 0 for image_id in image_ids) or len(image_ids) != len(set(image_ids)):
        raise ValueError(f"{value.retriever} vector mapping has invalid image IDs")
    return matrix, image_ids


def _new_index(index_type: str, dimension: int):  # noqa: ANN202
    if dimension <= 0:
        raise ValueError("index dimension must be positive")
    if index_type == "flat":
        return faiss.IndexFlatIP(dimension)
    if index_type == "hnsw":
        return faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    raise ValueError(f"unsupported index type: {index_type}")


def _describe(path: Path) -> BuiltFile:
    return BuiltFile(
        name=path.name,  # type: ignore[arg-type]
        path=str(path),
        sha256=_digest(path),
        length=path.stat().st_size,
    )
