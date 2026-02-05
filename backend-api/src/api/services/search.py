from typing import cast

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from activities.indexing import FaissIndexManager
from api.models.search import SearchResult
from shared.config import settings
from shared.models.orm import Annotation
from shared.models.orm import Image as ORMImage
from shared.services.storage import StorageService, get_storage_service


class SearchService:
    def __init__(self, index_manager: FaissIndexManager):
        self.index_manager = index_manager
        self._storage = cast(StorageService, get_storage_service())

    def search_by_embedding(
        self,
        embedding: list[float],
        db: Session,
        limit: int = 20,
    ) -> list[SearchResult]:
        if not self.index_manager.is_loaded:
            raise ValueError("Index not loaded")

        query_vector = np.array(embedding, dtype=np.float32)

        raw_results = self.index_manager.search(query_vector, k=limit)

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

        images = db.execute(select(ORMImage).where(ORMImage.id.in_(image_ids))).scalars().all()

        image_map = {img.id: img for img in images}

        annotations = (
            db.execute(select(Annotation).where(Annotation.image_id.in_(image_ids))).scalars().all()
        )

        annotation_map = {ann.image_id: ann for ann in annotations}

        results: list[SearchResult] = []

        for image_id, score in raw_results:
            img = image_map.get(image_id)
            if not img:
                continue

            ann = annotation_map.get(image_id)

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
                    caption=ann.caption if ann else None,
                    ocr_text=ann.ocr_text if ann else None,
                    width=img.width,
                    height=img.height,
                )
            )

        return results
