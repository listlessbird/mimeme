from __future__ import annotations

import json
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss  # type: ignore[import-untyped]
import numpy as np
from sqlalchemy.orm import Session

from shared.config import settings
from shared.models import IndexBuild
from shared.services.storage import get_storage_service


class FaissIndexManager:
    _instance: FaissIndexManager | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._cache_dir = settings.index_cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._storage = get_storage_service()
        self._index: faiss.Index | None = None
        self._id_mapping: dict[int, int] = {}
        self._reverse_mapping: dict[int, int] = {}
        self._active_version: str | None = None
        self._metadata: dict[str, Any] = {}
        self._index_lock = threading.RLock()

        # Text index (lazy-loaded from disk cache)
        self._text_index: faiss.Index | None = None
        self._text_id_mapping: dict[int, int] = {}
        self._text_metadata: dict[str, Any] = {}

    @classmethod
    def get_instance(cls) -> FaissIndexManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        assert cls._instance is not None
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._index is not None

    @property
    def is_text_loaded(self) -> bool:
        return self._text_index is not None

    @property
    def active_version(self) -> str | None:
        return self._active_version

    @property
    def num_vectors(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal

    @property
    def dimension(self) -> int | None:
        if self._index is None:
            return None
        return self._index.d

    def load_active_index(self, db: Session) -> None:
        with self._index_lock:
            active_build = db.query(IndexBuild).filter(IndexBuild.is_active).first()
            if active_build is None:
                raise FileNotFoundError("No active index found in db")
            self._load_index_version(active_build.version)

    def _load_index_version(self, version: str) -> None:
        cache_path = self._cache_dir / version
        index_file = cache_path / "index.faiss"
        mapping_file = cache_path / "mapping.json"
        metadata_file = cache_path / "metadata.json"

        if not index_file.exists():
            cache_path.mkdir(parents=True, exist_ok=True)

            index_key = self._storage.build_index_key(version, "index.faiss")
            mapping_key = self._storage.build_index_key(version, "mapping.json")
            metadata_key = self._storage.build_index_key(version, "metadata.json")

            download_jobs: list[tuple[str, Path]] = [(index_key, index_file), (mapping_key, mapping_file)]

            if self._storage.exists(metadata_key):
                download_jobs.append((metadata_key, metadata_file))

            # Also download text index files to disk cache (lazy – not loaded into memory)
            text_index_key = self._storage.build_index_key(version, "text_index.faiss")
            text_mapping_key = self._storage.build_index_key(version, "text_mapping.json")
            text_metadata_key = self._storage.build_index_key(version, "text_metadata.json")

            if self._storage.exists(text_index_key):
                download_jobs.append((text_index_key, cache_path / "text_index.faiss"))
                download_jobs.append((text_mapping_key, cache_path / "text_mapping.json"))
                if self._storage.exists(text_metadata_key):
                    download_jobs.append((text_metadata_key, cache_path / "text_metadata.json"))

            worker_count = min(8, len(download_jobs))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(self._storage.download_file, key, path)
                    for key, path in download_jobs
                ]
                for future in futures:
                    future.result()

        index = faiss.read_index(str(index_file))

        id_mapping: dict[int, int] = {}
        if mapping_file.exists():
            with open(mapping_file) as f:
                raw = json.load(f)
                id_mapping = {int(k): v for k, v in raw.items()}

        metadata: dict[str, Any] = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

        self._index = index
        self._id_mapping = id_mapping
        self._reverse_mapping = {v: k for k, v in id_mapping.items()}
        self._active_version = version
        self._metadata = metadata
        # Reset text index so it gets lazily reloaded for the new version
        self._text_index = None
        self._text_id_mapping = {}
        self._text_metadata = {}

    def search(self, query_vector: np.ndarray, k: int = 20) -> list[tuple[int, float]]:
        with self._index_lock:
            if self._index is None:
                raise ValueError("No index is loaded")

            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)

            query_vector = query_vector.astype(np.float32)
            faiss.normalize_L2(query_vector)

            distances, indices = self._index.search(query_vector, k)  # type: ignore[call-arg]

            results: list[tuple[int, float]] = []
            for idx, score in zip(indices[0], distances[0]):
                if idx < 0:
                    continue
                image_id = self._id_mapping.get(int(idx))
                if image_id is not None:
                    results.append((image_id, float(score)))

            return results

    def has_text_index(self) -> bool:
        """Check whether a text index exists on disk for the active version."""
        if self._text_index is not None:
            return True
        if self._active_version is None:
            return False
        cache_path = self._cache_dir / self._active_version
        return (cache_path / "text_index.faiss").exists()

    def ensure_text_index_loaded(self) -> None:
        """Lazily load the text FAISS index from disk cache into memory."""
        with self._index_lock:
            if self._text_index is not None:
                return
            if self._active_version is None:
                raise ValueError("No active index version")

            cache_path = self._cache_dir / self._active_version
            text_index_file = cache_path / "text_index.faiss"
            text_mapping_file = cache_path / "text_mapping.json"
            text_metadata_file = cache_path / "text_metadata.json"

            if not text_index_file.exists():
                raise FileNotFoundError(
                    f"Text index not found for version {self._active_version}"
                )

            self._text_index = faiss.read_index(str(text_index_file))

            if text_mapping_file.exists():
                with open(text_mapping_file) as f:
                    raw = json.load(f)
                    self._text_id_mapping = {int(k): v for k, v in raw.items()}

            if text_metadata_file.exists():
                with open(text_metadata_file) as f:
                    self._text_metadata = json.load(f)

    def search_text(self, query_vector: np.ndarray, k: int = 20) -> list[tuple[int, float]]:
        """Search the text FAISS index. Loads it lazily if not yet in memory."""
        with self._index_lock:
            self.ensure_text_index_loaded()
            assert self._text_index is not None

            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)

            query_vector = query_vector.astype(np.float32)
            faiss.normalize_L2(query_vector)

            distances, indices = self._text_index.search(query_vector, k)

            results: list[tuple[int, float]] = []
            for idx, score in zip(indices[0], distances[0]):
                if idx < 0:
                    continue
                image_id = self._text_id_mapping.get(int(idx))
                if image_id is not None:
                    results.append((image_id, float(score)))

            return results

    def get_vector_by_image_id(self, image_id: int) -> np.ndarray | None:
        with self._index_lock:
            if self._index is None:
                return None

            faiss_idx = self._reverse_mapping.get(image_id)
            if faiss_idx is None:
                return None

            return self._index.reconstruct(faiss_idx)  # type: ignore[call-arg]

    @staticmethod
    def _create_faiss_index(
        embeddings: np.ndarray,
        index_type: str,
        dimension: int,
        n_vectors: int,
    ) -> faiss.Index:
        if index_type == "flat":
            index = faiss.IndexFlatIP(dimension)
        elif index_type == "ivf":
            nlist = max(1, min(100, n_vectors // 10))
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)  # type: ignore[call-arg]
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        index.add(embeddings)  # type: ignore[call-arg]
        return index

    def build_index(
        self,
        embeddings: np.ndarray,
        image_ids: list[int],
        model_name: str,
        index_type: str = "flat",
        db: Session | None = None,
        text_embeddings: np.ndarray | None = None,
        text_image_ids: list[int] | None = None,
    ) -> str:
        if len(embeddings) != len(image_ids):
            raise ValueError("Embeddings and image_ids must have same length")

        n_vectors, dimension = embeddings.shape

        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        index = self._create_faiss_index(embeddings, index_type, dimension, n_vectors)

        version = f"v{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            faiss.write_index(index, str(tmp_path / "index.faiss"))

            mapping = {i: img_id for i, img_id in enumerate(image_ids)}
            with open(tmp_path / "mapping.json", "w") as f:
                json.dump(mapping, f)

            metadata = {
                "version": version,
                "model_name": model_name,
                "dimension": dimension,
                "num_vectors": n_vectors,
                "index_type": index_type,
                "created_at": datetime.now(UTC).isoformat(),
            }
            with open(tmp_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Build text index if text embeddings are provided
            if text_embeddings is not None and text_image_ids is not None:
                text_n = len(text_image_ids)
                text_dim = text_embeddings.shape[1]
                text_embs = text_embeddings.astype(np.float32)
                faiss.normalize_L2(text_embs)
                text_index = self._create_faiss_index(text_embs, index_type, text_dim, text_n)

                faiss.write_index(text_index, str(tmp_path / "text_index.faiss"))
                text_mapping = {i: img_id for i, img_id in enumerate(text_image_ids)}
                with open(tmp_path / "text_mapping.json", "w") as f:
                    json.dump(text_mapping, f)
                text_metadata = {
                    "version": version,
                    "model_name": model_name,
                    "dimension": text_dim,
                    "num_vectors": text_n,
                    "index_type": index_type,
                    "kind": "text",
                    "created_at": datetime.now(UTC).isoformat(),
                }
                with open(tmp_path / "text_metadata.json", "w") as f:
                    json.dump(text_metadata, f, indent=2)

            index_key = self._storage.build_index_key(version, "index.faiss")
            mapping_key = self._storage.build_index_key(version, "mapping.json")
            metadata_key = self._storage.build_index_key(version, "metadata.json")

            upload_jobs: list[tuple[Path, str]] = [
                (tmp_path / "index.faiss", index_key),
                (tmp_path / "mapping.json", mapping_key),
                (tmp_path / "metadata.json", metadata_key),
            ]

            # Include text index files in upload if they were built
            if (tmp_path / "text_index.faiss").exists():
                upload_jobs.extend([
                    (tmp_path / "text_index.faiss", self._storage.build_index_key(version, "text_index.faiss")),
                    (tmp_path / "text_mapping.json", self._storage.build_index_key(version, "text_mapping.json")),
                    (tmp_path / "text_metadata.json", self._storage.build_index_key(version, "text_metadata.json")),
                ])

            try:
                with ThreadPoolExecutor(max_workers=len(upload_jobs)) as executor:
                    futures = [
                        executor.submit(self._storage.upload_file, local_path, key)
                        for local_path, key in upload_jobs
                    ]
                    for future in futures:
                        future.result()
            except Exception:
                prefix = f"{self._storage.INDEXES_PREFIX}/{version}"
                for key, _ in self._storage.list_objects(prefix):
                    try:
                        self._storage.delete(key)
                    except Exception:
                        pass
                raise

            cache_path = self._cache_dir / version
            cache_path.mkdir(parents=True, exist_ok=True)
            for local_path, _ in upload_jobs:
                shutil.copy(local_path, cache_path / local_path.name)

        if db is not None:
            build_record = IndexBuild(
                version=version,
                s3_key=index_key,
                embed_model=model_name,
                index_type=index_type,
                num_vectors=n_vectors,
                dimension=dimension,
                is_active=False,
            )
            db.add(build_record)
            db.commit()

        return version

    def swap_to_version(self, version: str, db: Session) -> None:
        with self._index_lock:
            self._load_index_version(version)

            db.query(IndexBuild).filter(IndexBuild.is_active).update({"is_active": False})
            db.query(IndexBuild).filter(IndexBuild.version == version).update({"is_active": True})
            db.commit()

    def garbage_collect(self, db: Session, retain_versions: int = 5) -> list[str]:
        removed: list[str] = []

        all_builds = db.query(IndexBuild).order_by(IndexBuild.created_at.desc()).all()

        for build in all_builds[retain_versions:]:
            if build.is_active:
                continue

            if build.s3_key:
                prefix = f"{self._storage.INDEXES_PREFIX}/{build.version}"
                objects = self._storage.list_objects(prefix)
                if objects:
                    with ThreadPoolExecutor(max_workers=min(8, len(objects))) as executor:
                        futures = [executor.submit(self._storage.delete, key) for key, _ in objects]
                        for future in futures:
                            future.result()

            cache_path = self._cache_dir / build.version
            if cache_path.exists():
                shutil.rmtree(cache_path)

            db.delete(build)
            removed.append(build.version)

        db.commit()
        return removed
