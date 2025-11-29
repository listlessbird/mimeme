import json
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import structlog

log = structlog.get_logger()


class FaissIndexManager:
    def __init__(self, index_dir: Path, db_url: str, retain_versions: int = 5) -> None:
        self.index_dir = Path(index_dir)
        self.db_url = db_url
        self.retain_versions = retain_versions

        self._index: faiss.Index | None = None
        self._id_mapping: dict[int, int] = {}  # faiss_idx -> image_id
        self._reverse_mapping: dict[int, int] = {}
        self._active_version: str | None = None
        self._metadata: dict[str, Any] = {}

        self._lock = threading.RLock()

        # prolly not needed, since its already ensured in lifecycle, but w/e
        self.index_dir.mkdir(parents=True, exist_ok=True)

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

    def load_active_index(self) -> None:
        with self._lock:
            current_link = self.index_dir / "current"

            if current_link.is_symlink():
                version_dir = current_link.resolve()

            else:
                versions = sorted(
                    [d for d in self.index_dir.iterdir() if d.is_dir() and d.name.startswith("v")],
                    reverse=True,
                )
                if not versions:
                    raise FileNotFoundError("No index versions found")
                version_dir = versions[0]
            self._load_index_from_dir(version_dir)

    def _load_index_from_dir(self, version_dir: Path) -> None:
        index_path = version_dir / "index.faiss"
        mapping_path = version_dir / "mapping.json"
        metadata_path = version_dir / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        index = faiss.read_index(str(index_path))

        id_mapping = {}
        if mapping_path.exists():
            with open(mapping_path) as f:
                raw_mapping = json.load(f)
                id_mapping = {int(k): v for k, v in raw_mapping.items()}

        metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        self._index = index
        self._id_mapping = id_mapping
        self._reverse_mapping = {v: k for k, v in id_mapping.items()}
        self._active_version = version_dir.name
        self._metadata = metadata

        log.info(
            "faiss index loaded",
            version=self._active_version,
            num_vectors=index.ntotal,
            dimension=index.d,
        )

    def search(self, query_vector: np.ndarray, k: int = 20) -> list[tuple[int, float]]:
        with self._lock:
            if self._index is None:
                raise ValueError("No index is loaded")

            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)

            query_vector = query_vector.astype(np.float32)
            faiss.normalize_L2(query_vector)

            distances, indicies = self._index.search(query_vector, k)  # type: ignore[call-arg]

            results = []

            for idx, score in zip(indicies[0], distances[0]):
                if idx < 0:
                    continue
                image_id = self._id_mapping.get(int(idx))
                if image_id is not None:
                    results.append((image_id, float(score)))

            return results

    def get_vector_by_image_id(self, image_id: int) -> np.ndarray | None:
        with self._lock:
            if self._index is None:
                return None

            faiss_idx = self._reverse_mapping.get(image_id)
            if faiss_idx is None:
                return None

            return self._index.reconstruct(faiss_idx)  # type: ignore[call-arg]

    def swap_to_version(self, version: str) -> None:
        version_dir = self.index_dir / version

        if not version_dir.exists():
            raise FileNotFoundError(f"Index version not found: {version}")

        with self._lock:
            self._load_index_from_dir(version_dir)

            current_link = self.index_dir / "current"
            temp_link = self.index_dir / "current.tmp"

            temp_link.symlink_to(version_dir)
            temp_link.replace(current_link)

        log.info("index_swapped", new_version=version)

    def build_index(
        self,
        embeddings: np.ndarray,
        image_ids: list[int],
        model_name: str,
        index_type: str = "flat",
    ) -> str:
        if len(embeddings) != len(image_ids):
            raise ValueError("Embeddings and image_ids must have same length")

        n_vectors, dimension = embeddings.shape

        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        if index_type == "flat":
            index = faiss.IndexFlatIP(dimension)
        elif index_type == "ivf":
            nlist = min(100, n_vectors // 10)  # around 10 vct per cluster
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)  # type: ignore[call-arg]

        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)

        else:
            raise ValueError(f"Unknown index type: {index_type}")

        index.add(embeddings)  # type: ignore[call-arg]

        version = f"v{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        version_dir = self.index_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(version_dir / "index.faiss"))

        mapping = {i: img_id for i, img_id in enumerate(image_ids)}
        with open(version_dir / "mapping.json", "w") as f:
            json.dump(mapping, f)

        metadata = {
            "version": version,
            "model_name": model_name,
            "dimension": dimension,
            "num_vectors": n_vectors,
            "index_type": index_type,
            "created_at": datetime.now(UTC).isoformat(),
        }

        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        log.info(
            "index_built", version=version, num_vectors=n_vectors, dim=dimension, type=index_type
        )

        return version

    def garbage_collect(self) -> list[str]:
        versions = sorted(
            [d for d in self.index_dir.iterdir() if d.is_dir() and d.name.startswith("v")],
            reverse=True,
        )

        removed = []

        for version_dir in versions[self.retain_versions :]:
            if version_dir.name == self.active_version:
                continue

            shutil.rmtree(version_dir)
            removed.append(version_dir.name)
            log.info("index_version_removed", version=version_dir.name)

        return removed
