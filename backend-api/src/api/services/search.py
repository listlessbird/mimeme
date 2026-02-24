from __future__ import annotations

from typing import Literal, cast

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from activities.indexing import FaissIndexManager
from api.models.search import SearchResult
from shared.config import settings
from shared.models.orm import Annotation
from shared.models.orm import Image as ORMImage
from shared.services.storage import StorageService, get_storage_service


def reciprocal_rank_fusion(
    image_results: list[tuple[int, float]],
    text_results: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Merge two ranked result lists using Reciprocal Rank Fusion.

    For each result, score = 1 / (rank + k), summed across both lists.
    """
    scores: dict[int, float] = {}
    for rank, (image_id, _) in enumerate(image_results, start=1):
        scores[image_id] = scores.get(image_id, 0) + 1 / (rank + k)
    for rank, (image_id, _) in enumerate(text_results, start=1):
        scores[image_id] = scores.get(image_id, 0) + 1 / (rank + k)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class SearchService:
    def __init__(self, index_manager: FaissIndexManager):
        self.index_manager = index_manager
        self._storage = cast(StorageService, get_storage_service())

    def search_by_embedding(
        self,
        embedding: list[float],
        db: Session,
        limit: int = 20,
        mode: Literal["image", "text", "hybrid"] = "hybrid",
    ) -> list[SearchResult]:
        if not self.index_manager.is_loaded:
            raise ValueError("Index not loaded")

        query_vector = np.array(embedding, dtype=np.float32)

        if mode == "image":
            raw_results = self.index_manager.search(query_vector, k=limit)
        elif mode == "text":
            if not self.index_manager.has_text_index():
                raise ValueError("Text index not available")
            raw_results = self.index_manager.search_text(query_vector, k=limit)
        else:
            # hybrid: use RRF to merge image and text results
            image_results = self.index_manager.search(query_vector, k=limit)
            if self.index_manager.has_text_index():
                text_results = self.index_manager.search_text(query_vector, k=limit)
                raw_results = reciprocal_rank_fusion(image_results, text_results)[:limit]
            else:
                # Fall back to image-only if no text index exists
                raw_results = image_results

        if not raw_results:
            return []

        return self._hydrate_results(raw_results, db)

    def find_similar(
        self,
        image_id: int,
        db: Session,
        limit: int = 20,
    ) -> list[SearchResult]:
        query_vector = self.index_manager.get_vector_by_image_id(image_id)

        if query_vector is None:
            raise ValueError(f"No embedding found for image ID {image_id}")

        raw_results = self.index_manager.search(query_vector, k=limit + 1)

        return self._hydrate_results(raw_results, db)

    def _hydrate_results(
        self, raw_results: list[tuple[int, float]], db: Session
    ) -> list[SearchResult]:
        image_ids = [r[0] for r in raw_results]

        image_rows = db.execute(
            select(ORMImage, Annotation)
            .outerjoin(Annotation, Annotation.image_id == ORMImage.id)
            .where(ORMImage.id.in_(image_ids))
        ).all()

        image_map = {img.id: (img, ann) for img, ann in image_rows}

        results: list[SearchResult] = []

        for image_id, score in raw_results:
            row = image_map.get(image_id)
            if row is None:
                continue

            img, ann = row

            url = None

            if img.s3_key:
                url = self._storage.generate_presigned_url(
                    img.s3_key, expiration=settings.s3_presigned_url_expiry
                )

            results.append(
                SearchResult(
                    id=img.id,
                    sha256=img.sha256,
                    score=score,
                    url=url,
                    caption=ann.caption_text if ann else None,
                    ocr_text=ann.ocr_text if ann else None,
                    width=img.width,
                    height=img.height,
                )
            )

        return results
