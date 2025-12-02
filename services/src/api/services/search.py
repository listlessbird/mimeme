import select
import threading

import numpy as np
import structlog
import torch
from sqlalchemy import select
from sqlalchemy.orm import Session
from transformers import AutoModel, AutoProcessor

from api.config import settings
from api.models.orm import Annotation
from api.models.orm import Image as ORMImage
from api.models.search import SearchResult
from api.services.indexer import FaissIndexManager
from api.services.storage import get_storage_service

log = structlog.get_logger()


class SearchService:
    def __init__(
        self,
        index_manager: FaissIndexManager,
        model_name: str = "google/siglip2-base-patch16-naflex",
        device: str = "cuda",
    ) -> None:
        self.index_manager = index_manager
        self.model_name = model_name
        self.device = device

        self._model: AutoModel | None = None
        self._processor: AutoProcessor | None = None
        self._model_lock = threading.Lock()

        self._is_siglip2 = "siglip2" in model_name.lower()
        self._storage = get_storage_service()

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        with self._model_lock:
            if self._model is not None:
                return

            log.info("loading_text_encoder", model=self.model_name)

            self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

            self._model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None,
                # dtype=(torch.float16 if dtype == torch.float16 else None),
                dtype=torch.float16,
            )

            log.info("text_encoder_loaded")

    def encode_query(self, query: str) -> np.ndarray:
        self._ensure_model_loaded()

        if self._is_siglip2:
            query = query.lower()

        # tokenize
        inputs = self._processor(  # type: ignore[misc]
            text=[query],
            return_tensors="pt",
            padding="max_length",
            max_length=64 if self._is_siglip2 else 77,
        )

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}  # type: ignore[union-attr]

        with torch.no_grad():
            if hasattr(self._model, "get_text_features"):
                features = self._model.get_text_features(**inputs)  # type: ignore[attr-defined]
            else:
                outputs = self._model(**inputs)  # type: ignore[misc]
                if hasattr(outputs, "text_embeds"):
                    features = outputs.text_embeds
                elif hasattr(outputs, "pooler_output"):
                    features = outputs.pooler_output
                else:
                    features = outputs.last_hidden_state[:, 0, :]

        embedding = features.cpu().numpy().squeeze()

        return embedding

    def search(
        self,
        query: str,
        limit: int = 10,
        db: Session | None = None,
    ) -> list[SearchResult]:
        if not self.index_manager.is_loaded:
            raise ValueError("No search index is loaded")

        query_vector = self.encode_query(query)

        raw_results = self.index_manager.search(query_vector, limit)

        if not raw_results:
            return []

        if db is None:
            return [
                SearchResult(
                    id=image_id,
                    sha256="",
                    score=score,
                    caption="",
                    height=None,
                    ocr_text=None,
                    url=None,
                    width=None,
                )
                for image_id, score in raw_results
            ]

        image_ids = [r[0] for r in raw_results]

        images = db.execute(select(ORMImage).where(ORMImage.id.in_(image_ids))).scalars().all()

        image_map = {img.id: img for img in images}

        annotations = (
            db.execute(select(Annotation).where(Annotation.image_id.in_(image_ids))).scalars().all()
        )

        annotation_map = {ann.image_id: ann for ann in annotations}

        results = []

        for image_id, score in raw_results:
            img = image_map.get(image_id)
            if img is None:
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
                    caption=ann.caption_text if ann else None,
                    ocr_text=ann.ocr_text if ann else None,
                    width=img.width,
                    height=img.height,
                )
            )

        return results

    def find_similar(
        self, image_id: int, limit: int = 20, db: Session | None = None
    ) -> list[SearchResult]:
        query_vector = self.index_manager.get_vector_by_image_id(image_id)

        if query_vector is None:
            raise ValueError(f"Image {image_id} not found in index")

        raw_results = self.index_manager.search(query_vector, limit)

        if db is None:
            return [
                SearchResult(
                    id=image_id,
                    sha256="",
                    score=score,
                    caption="",
                    height=None,
                    ocr_text=None,
                    url=None,
                    width=None,
                )
                for img_id, score in raw_results
            ]

        image_ids = [r[0] for r in raw_results]

        images = db.execute(select(ORMImage).where(ORMImage.id.in_(image_ids))).scalars().all()

        image_map = {img.id: img for img in images}

        annotations = (
            db.execute(select(Annotation).where(Annotation.image_id.in_(image_ids))).scalars().all()
        )

        annotation_map = {ann.image_id: ann for ann in annotations}

        results = []

        for image_id, score in raw_results:
            img = image_map.get(image_id)
            if img is None:
                continue

            ann = annotation_map.get(image_id)

            results.append(
                SearchResult(
                    id=img.id,
                    sha256=img.sha256,
                    score=score,
                    url=None,
                    caption=ann.caption_text if ann else None,
                    ocr_text=ann.ocr_text if ann else None,
                    width=img.width,
                    height=img.height,
                )
            )

        return results
