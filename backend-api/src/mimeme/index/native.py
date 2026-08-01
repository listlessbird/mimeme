"""Blocking NumPy and FAISS index construction owned by the index child."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import faiss  # type: ignore[import-untyped]
import numpy as np
import onnxruntime as ort

from mimeme.index.model import Built, BuiltFile, PreparedBuild


def build(request: PreparedBuild) -> Built:
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


def _load(request: PreparedBuild, *, text: bool) -> tuple[np.ndarray, list[int]]:
    vectors: list[np.ndarray] = []
    image_ids: list[int] = []
    for item in request.embeddings:
        raw_path = item.text_path if text else item.image_path
        if raw_path is None:
            continue
        vector = np.asarray(np.load(raw_path, allow_pickle=False), dtype=np.float32)
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            raise ValueError(f"invalid embedding for image {item.image_id}")
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError(f"zero embedding for image {item.image_id}")
        vectors.append(vector / norm)
        image_ids.append(item.image_id)
    if not vectors:
        return np.empty((0, request.dimension), dtype=np.float32), []
    return np.stack(vectors).astype(np.float32, copy=False), image_ids


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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return BuiltFile(
        name=path.name,  # type: ignore[arg-type]
        path=str(path),
        sha256=digest.hexdigest(),
        length=path.stat().st_size,
    )
