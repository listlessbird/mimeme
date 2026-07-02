from __future__ import annotations

import numpy as np


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def cosine_diagonal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(l2_normalize(a) * l2_normalize(b), axis=-1)


def pairwise_cosine(embeddings: np.ndarray) -> np.ndarray:
    normalized = l2_normalize(embeddings)
    return normalized @ normalized.T


def top_k_neighbors(similarity: np.ndarray, k: int) -> np.ndarray:
    masked = similarity.copy()
    np.fill_diagonal(masked, -np.inf)
    return np.argsort(-masked, axis=1)[:, :k]


def neighbor_overlap(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    overlaps = [
        len(set(ref_row.tolist()) & set(cand_row.tolist()))
        for ref_row, cand_row in zip(reference, candidate)
    ]
    return np.array(overlaps)
