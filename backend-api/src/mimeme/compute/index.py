"""Blocking NumPy and FAISS construction owned by the index compute child."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import faiss  # type: ignore[import-untyped]
import numpy as np
import onnxruntime as ort

from mimeme.index.model import (
    Built,
    BuiltFile,
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

    return Built(
        version=request.version,
        target_generation=request.target_generation,
        model=request.model,
        index_type=request.index_type,
        dimension=request.dimension,
        image_count=len(image_ids),
        text_count=len(text_ids) or None,
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
    kind: str,
) -> list[Path]:
    index = _new_index(request.index_type, request.dimension)
    if len(image_ids):
        index.add(vectors)  # type: ignore[call-arg]
    prefix = "text_" if kind == "text" else ""
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
                "embed_model": request.model,
                "dimension": request.dimension,
                "metric": "inner_product",
                "normalized": True,
                "faiss_version": faiss.__version__,
                "onnxruntime_version": ort.__version__,
                "encoder_repo": request.encoder.repo,
                "encoder_revision": request.encoder.revision,
                "encoder_variant": request.encoder.variant,
                "kind": kind,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return [index_path, mapping_path, metadata_path]


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
