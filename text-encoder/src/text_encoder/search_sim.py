from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class SnapshotIndex:
    def __init__(self, index_file: Path, mapping_file: Path) -> None:
        self.index = faiss.read_index(str(index_file))
        mapping = json.loads(mapping_file.read_text())
        self.id_mapping = {int(row): image_id for row, image_id in mapping.items()}

    def search(self, query_vectors: np.ndarray, k: int) -> list[list[tuple[int, float]]]:
        queries = query_vectors.astype(np.float32).copy()
        faiss.normalize_L2(queries)
        distances, indices = self.index.search(queries, k)
        results = []
        for row_indices, row_scores in zip(indices, distances):
            hits = []
            for row, score in zip(row_indices, row_scores):
                if row < 0:
                    continue
                image_id = self.id_mapping.get(int(row))
                if image_id is not None:
                    hits.append((image_id, float(score)))
            results.append(hits)
        return results


def reciprocal_rank_fusion(
    image_results: list[tuple[int, float]],
    text_results: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rank, (image_id, _) in enumerate(image_results, start=1):
        scores[image_id] = scores.get(image_id, 0) + 1 / (rank + k)
    for rank, (image_id, _) in enumerate(text_results, start=1):
        scores[image_id] = scores.get(image_id, 0) + 1 / (rank + k)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class SearchSnapshot:
    def __init__(self, snapshot_dir: Path) -> None:
        self.version = json.loads((snapshot_dir / "metadata.json").read_text())["version"]
        self.image_index = SnapshotIndex(
            snapshot_dir / "index.faiss", snapshot_dir / "mapping.json"
        )
        self.text_index = SnapshotIndex(
            snapshot_dir / "text_index.faiss", snapshot_dir / "text_mapping.json"
        )
        self._exact: dict[str, tuple[np.ndarray, dict[int, int]]] | None = None

    def search_all(self, query_vectors: np.ndarray, k: int) -> dict[str, list[list[int]]]:
        image = self.image_index.search(query_vectors, k)
        text = self.text_index.search(query_vectors, k)
        return self._fuse(image, text, k)

    def exact_search_all(self, query_vectors: np.ndarray, k: int) -> dict[str, list[list[int]]]:
        if self._exact is None:
            self._exact = {
                "image": (self._reconstruct(self.image_index), self.image_index.id_mapping),
                "text": (self._reconstruct(self.text_index), self.text_index.id_mapping),
            }
        queries = query_vectors.astype(np.float32).copy()
        faiss.normalize_L2(queries)
        results = {}
        for kind, (base, mapping) in self._exact.items():
            top = np.argsort(-(queries @ base.T), axis=1)[:, :k]
            results[kind] = [[(mapping[int(row)], 0.0) for row in query_rows] for query_rows in top]
        return self._fuse(results["image"], results["text"], k)

    @staticmethod
    def _reconstruct(index: SnapshotIndex) -> np.ndarray:
        return np.vstack([index.index.reconstruct(i) for i in range(index.index.ntotal)])

    @staticmethod
    def _fuse(
        image: list[list[tuple[int, float]]],
        text: list[list[tuple[int, float]]],
        k: int,
    ) -> dict[str, list[list[int]]]:
        hybrid = [
            [image_id for image_id, _ in reciprocal_rank_fusion(img, txt)[:k]]
            for img, txt in zip(image, text)
        ]
        return {
            "image": [[image_id for image_id, _ in hits] for hits in image],
            "hybrid": hybrid,
        }
