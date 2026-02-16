from __future__ import annotations

import json
import shutil
import tempfile
import threading
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

            worker_count = min(4, len(download_jobs))
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

    def get_vector_by_image_id(self, image_id: int) -> np.ndarray | None:
        with self._index_lock:
            if self._index is None:
                return None

            faiss_idx = self._reverse_mapping.get(image_id)
            if faiss_idx is None:
                return None

            return self._index.reconstruct(faiss_idx)  # type: ignore[call-arg]

    def build_index(
        self,
        embeddings: np.ndarray,
        image_ids: list[int],
        model_name: str,
        index_type: str = "flat",
        db: Session | None = None,
    ) -> str:
        if len(embeddings) != len(image_ids):
            raise ValueError("Embeddings and image_ids must have same length")

        n_vectors, dimension = embeddings.shape

        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        if index_type == "flat":
            index = faiss.IndexFlatIP(dimension)
        elif index_type == "ivf":
            # FAISS requires nlist >= 1; very small datasets (<10 vectors)
            # previously produced nlist=0 and raised at index construction time.
            nlist = max(1, min(100, n_vectors // 10))
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)  # type: ignore[call-arg]
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        index.add(embeddings)  # type: ignore[call-arg]

        version = f"v{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"

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

            index_key = self._storage.build_index_key(version, "index.faiss")
            mapping_key = self._storage.build_index_key(version, "mapping.json")
            metadata_key = self._storage.build_index_key(version, "metadata.json")

            upload_jobs = [
                (tmp_path / "index.faiss", index_key),
                (tmp_path / "mapping.json", mapping_key),
                (tmp_path / "metadata.json", metadata_key),
            ]
            with ThreadPoolExecutor(max_workers=len(upload_jobs)) as executor:
                futures = [
                    executor.submit(self._storage.upload_file, local_path, key)
                    for local_path, key in upload_jobs
                ]
                for future in futures:
                    future.result()

            cache_path = self._cache_dir / version
            cache_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(tmp_path / "index.faiss", cache_path / "index.faiss")
            shutil.copy(tmp_path / "mapping.json", cache_path / "mapping.json")
            shutil.copy(tmp_path / "metadata.json", cache_path / "metadata.json")

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
