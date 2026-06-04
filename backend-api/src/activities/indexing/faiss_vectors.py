from __future__ import annotations

from pathlib import Path

import faiss  # type: ignore[import-untyped]
import numpy as np


class FaissVectorIndex:
    def __init__(self, index: faiss.Index, id_mapping: dict[int, int]) -> None:
        self.index = index
        self.id_mapping = id_mapping
        self.reverse_mapping = {image_id: row for row, image_id in id_mapping.items()}

    @classmethod
    def build(
        cls,
        *,
        embeddings: np.ndarray,
        image_ids: list[int],
        index_type: str,
    ) -> FaissVectorIndex:
        if len(embeddings) != len(image_ids):
            raise ValueError("Embeddings and image_ids must have same length")

        n_vectors, dimension = embeddings.shape
        normalized = normalize_embeddings(embeddings)
        index = create_faiss_index(
            embeddings=normalized,
            index_type=index_type,
            dimension=dimension,
            n_vectors=n_vectors,
        )
        return cls(index=index, id_mapping={i: image_id for i, image_id in enumerate(image_ids)})

    @classmethod
    def read(cls, *, index_file: Path, id_mapping: dict[int, int]) -> FaissVectorIndex:
        return cls(index=faiss.read_index(str(index_file)), id_mapping=id_mapping)

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    @property
    def dimension(self) -> int:
        return self.index.d

    def write(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        query = normalize_query(query_vector)
        distances, indices = self.index.search(query, k)  # type: ignore[call-arg]

        results: list[tuple[int, float]] = []
        for row, score in zip(indices[0], distances[0]):
            if row < 0:
                continue
            image_id = self.id_mapping.get(int(row))
            if image_id is not None:
                results.append((image_id, float(score)))
        return results

    def get_vector_by_image_id(self, image_id: int) -> np.ndarray | None:
        faiss_row = self.reverse_mapping.get(image_id)
        if faiss_row is None:
            return None
        return self.index.reconstruct(faiss_row)  # type: ignore[call-arg]


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    normalized = embeddings.astype(np.float32)
    faiss.normalize_L2(normalized)
    return normalized


def normalize_query(query_vector: np.ndarray) -> np.ndarray:
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    query = query_vector.astype(np.float32)
    faiss.normalize_L2(query)
    return query


def create_faiss_index(
    *,
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
