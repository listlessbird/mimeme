from __future__ import annotations

import tempfile
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import structlog
from shared.config import settings
from shared.services.storage import get_storage_service
from sqlalchemy.orm import Session

from activities.indexing.faiss_vectors import FaissVectorIndex
from activities.indexing.index_artifacts import IndexArtifactStore
from activities.indexing.index_catalog import ActiveIndexCatalog

log = structlog.get_logger()


class BuildResult(NamedTuple):
    version: str
    text_num_vectors: int | None
    text_s3_key: str | None


class FaissIndexManager:
    _instance: FaissIndexManager | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._storage = get_storage_service()
        self._artifacts = IndexArtifactStore(
            storage=self._storage,
            cache_dir=settings.index_cache_dir,
        )
        self._catalog = ActiveIndexCatalog()
        self._index_lock = threading.RLock()

        self._index: FaissVectorIndex | None = None
        self._active_version: str | None = None
        self._metadata: dict[str, Any] = {}

        self._text_index: FaissVectorIndex | None = None
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
        return self._index.dimension

    def load_active_index(self, db: Session) -> None:
        with self._index_lock:
            active_build = self._catalog.active_build(db)
            if active_build is None:
                raise FileNotFoundError("No active index found in db")
            self._load_index_version(active_build.version)

    def load_index_version(self, version: str) -> None:
        with self._index_lock:
            self._load_index_version(version)

    @staticmethod
    def _version_sort_key(version: str) -> tuple[int, str]:
        return IndexArtifactStore.version_sort_key(version)

    def list_available_versions(self) -> list[str]:
        return self._artifacts.list_versions()

    def _has_required_artifacts(self, version: str) -> bool:
        return self._artifacts.has_required_artifacts(version)

    def autoload_latest_available(self, db: Session) -> str | None:
        versions = [
            version
            for version in self._artifacts.list_versions()
            if self._artifacts.has_required_artifacts(version)
        ]
        if not versions:
            return None

        latest = versions[-1]
        if self._active_version == latest and self._index is not None:
            return None

        with self._index_lock:
            self._load_index_version(latest)
            self._catalog.mark_latest_available(
                db,
                version=latest,
                metadata=self._metadata,
                artifacts=self._artifacts,
            )

        return latest

    def _load_index_version(self, version: str) -> None:
        paths = self._artifacts.ensure_cached(version)
        index = FaissVectorIndex.read(
            index_file=paths.index_file,
            id_mapping=self._artifacts.read_mapping(paths.mapping_file),
        )
        metadata = self._artifacts.read_metadata(paths.metadata_file)

        self._index = index
        self._active_version = version
        self._metadata = metadata
        self._text_index = None
        self._text_metadata = {}
        log.info(
            "index_load_step",
            step="complete",
            version=version,
            ntotal=index.ntotal,
            dimension=index.dimension,
            has_text_index=self._artifacts.has_text_artifacts_cached(version),
        )

    def search(self, query_vector: np.ndarray, k: int = 20) -> list[tuple[int, float]]:
        with self._index_lock:
            if self._index is None:
                raise ValueError("No index is loaded")
            return self._index.search(query_vector, k)

    def has_text_index(self) -> bool:
        if self._text_index is not None:
            return True
        if self._active_version is None:
            return False
        return self._artifacts.has_text_artifacts_cached(self._active_version)

    def ensure_text_index_loaded(self) -> None:
        with self._index_lock:
            if self._text_index is not None:
                return
            if self._active_version is None:
                raise ValueError("No active index version")

            paths = self._artifacts.paths(self._active_version)
            if not paths.text_index_file.exists():
                raise FileNotFoundError(f"Text index not found for version {self._active_version}")

            self._text_index = FaissVectorIndex.read(
                index_file=paths.text_index_file,
                id_mapping=self._artifacts.read_mapping(paths.text_mapping_file),
            )
            self._text_metadata = self._artifacts.read_metadata(paths.text_metadata_file)
            log.info(
                "text_index_load_step",
                step="complete",
                version=self._active_version,
                ntotal=self._text_index.ntotal,
                dimension=self._text_index.dimension,
                metadata_keys=sorted(self._text_metadata.keys()),
            )

    def search_text(self, query_vector: np.ndarray, k: int = 20) -> list[tuple[int, float]]:
        with self._index_lock:
            self.ensure_text_index_loaded()
            assert self._text_index is not None
            return self._text_index.search(query_vector, k)

    def get_vector_by_image_id(self, image_id: int) -> np.ndarray | None:
        with self._index_lock:
            if self._index is None:
                return None
            return self._index.get_vector_by_image_id(image_id)

    @staticmethod
    def _create_faiss_index(
        embeddings: np.ndarray,
        index_type: str,
        dimension: int,
        n_vectors: int,
    ):
        from activities.indexing.faiss_vectors import create_faiss_index

        return create_faiss_index(
            embeddings=embeddings,
            index_type=index_type,
            dimension=dimension,
            n_vectors=n_vectors,
        )

    def build_index(
        self,
        embeddings: np.ndarray,
        image_ids: list[int],
        model_name: str,
        index_type: str = "flat",
        db: Session | None = None,
        text_embeddings: np.ndarray | None = None,
        text_image_ids: list[int] | None = None,
    ) -> BuildResult:
        image_index = FaissVectorIndex.build(
            embeddings=embeddings,
            image_ids=image_ids,
            index_type=index_type,
        )
        n_vectors, dimension = embeddings.shape
        version = f"v{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_index.write(tmp_path / "index.faiss")
            self._artifacts.write_json(tmp_path / "mapping.json", image_index.id_mapping)
            self._artifacts.write_json(
                tmp_path / "metadata.json",
                {
                    "version": version,
                    "model_name": model_name,
                    "dimension": dimension,
                    "num_vectors": n_vectors,
                    "index_type": index_type,
                    "created_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
            )

            text_index = self._build_text_index(
                tmp_path=tmp_path,
                version=version,
                text_embeddings=text_embeddings,
                text_image_ids=text_image_ids,
                model_name=model_name,
                index_type=index_type,
            )

            local_paths = sorted(tmp_path.iterdir())
            self._artifacts.upload_files(version, local_paths)
            self._artifacts.copy_to_cache(version, local_paths)

            text_s3_key = (
                self._artifacts.key(version, "text_index.faiss") if text_index is not None else None
            )
            text_num_vectors = (
                len(text_image_ids)
                if text_index is not None and text_image_ids is not None
                else None
            )

        if db is not None:
            self._catalog.add_inactive_build(
                db,
                version=version,
                s3_key=self._artifacts.key(version, "index.faiss"),
                embed_model=model_name,
                index_type=index_type,
                num_vectors=n_vectors,
                dimension=dimension,
            )

        return BuildResult(
            version=version,
            text_num_vectors=text_num_vectors,
            text_s3_key=text_s3_key,
        )

    def swap_to_version(self, version: str, db: Session) -> None:
        with self._index_lock:
            self._load_index_version(version)
            self._catalog.swap_to_version(version, db)

    def garbage_collect(self, db: Session, retain_versions: int = 5) -> list[str]:
        return self._catalog.garbage_collect(
            db,
            artifacts=self._artifacts,
            retain_versions=retain_versions,
        )

    def _build_text_index(
        self,
        *,
        tmp_path: Path,
        version: str,
        text_embeddings: np.ndarray | None,
        text_image_ids: list[int] | None,
        model_name: str,
        index_type: str,
    ) -> FaissVectorIndex | None:
        if text_embeddings is None or text_image_ids is None:
            return None

        text_index = FaissVectorIndex.build(
            embeddings=text_embeddings,
            image_ids=text_image_ids,
            index_type=index_type,
        )
        text_index.write(tmp_path / "text_index.faiss")
        self._artifacts.write_json(tmp_path / "text_mapping.json", text_index.id_mapping)
        self._artifacts.write_json(
            tmp_path / "text_metadata.json",
            {
                "version": version,
                "model_name": model_name,
                "dimension": text_embeddings.shape[1],
                "num_vectors": len(text_image_ids),
                "index_type": index_type,
                "kind": "text",
                "created_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        return text_index
