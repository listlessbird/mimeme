from __future__ import annotations

import time
from time import perf_counter
from typing import Literal, cast

import numpy as np
import structlog
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from activities.indexing import FaissIndexManager
from api.models.search import SearchResult
from api.services.text_encoder import SearchTextEncoder
from shared.config import settings
from shared.db import read_session_scope
from shared.models import IndexBuild
from shared.models.orm import Annotation
from shared.models.orm import Image as ORMImage
from shared.services.storage import StorageService, get_storage_service

log = structlog.get_logger()

_last_index_check: float = 0.0
_INDEX_CHECK_INTERVAL: float = 60.0
_active_embed_model: str | None = None


class SearchIndexUnavailableError(Exception):
    """Raised when there is no active search index available."""


class SearchIndexLoadError(Exception):
    """Raised when an active search index exists but cannot be loaded."""


class SearchQueryEncodingError(Exception):
    """Raised when a text query cannot be encoded."""


class SearchEncoderIncompatibleError(Exception):
    """Raised when the text encoder's source model does not match the active index."""


class SearchInvalidRequestError(Exception):
    """Raised when search input is invalid for the loaded index."""


class SearchImageNotFoundError(Exception):
    """Raised when similar search cannot find the query image vector."""


class SearchExecutionFailedError(Exception):
    """Raised when search execution fails unexpectedly."""


class SearchIndexPage(BaseModel, frozen=True):
    query: str
    results: list[SearchResult]
    total: int
    limit: int
    offset: int
    search_time_ms: float
    index_version: str | None


def check_encoder_index_compatibility(encoder: object, index_embed_model: str | None) -> None:
    encoder_model = getattr(encoder, "source_model", None)
    if encoder_model and index_embed_model and encoder_model != index_embed_model:
        log.error(
            "text_encoder_index_model_mismatch",
            encoder_source_model=encoder_model,
            index_embed_model=index_embed_model,
        )
        raise SearchEncoderIncompatibleError(
            f"Text encoder was exported from {encoder_model!r} but the active "
            f"index was built with {index_embed_model!r}"
        )


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


class SearchService:
    def __init__(
        self,
        index_manager: FaissIndexManager,
        storage: StorageService | None = None,
    ) -> None:
        self.index_manager = index_manager
        self._storage = storage or cast(StorageService, get_storage_service())

    def search_by_embedding(
        self,
        embedding: list[float],
        db: Session,
        limit: int = 20,
        mode: Literal["image", "hybrid"] = "hybrid",
    ) -> list[SearchResult]:
        if not self.index_manager.is_loaded:
            raise ValueError("Index not loaded")

        query_vector = np.array(embedding, dtype=np.float32)
        has_text_index = self.index_manager.has_text_index()
        log.info(
            "search_execution_plan",
            mode=mode,
            requested_limit=limit,
            index_version=self.index_manager.active_version,
            has_text_index=has_text_index,
            is_text_loaded=self.index_manager.is_text_loaded,
        )

        if mode == "image":
            started = perf_counter()
            raw_results = self.index_manager.search(query_vector, k=limit)
            log.info(
                "search_image_done",
                mode=mode,
                requested_limit=limit,
                candidate_count=len(raw_results),
                duration_ms=int((perf_counter() - started) * 1000),
                index_version=self.index_manager.active_version,
            )
        else:
            image_started = perf_counter()
            image_results = self.index_manager.search(query_vector, k=limit)
            image_search_ms = int((perf_counter() - image_started) * 1000)
            if has_text_index:
                text_started = perf_counter()
                text_results = self.index_manager.search_text(query_vector, k=limit)
                text_search_ms = int((perf_counter() - text_started) * 1000)
                fusion_started = perf_counter()
                raw_results = reciprocal_rank_fusion(image_results, text_results)[:limit]
                fusion_ms = int((perf_counter() - fusion_started) * 1000)
                log.info(
                    "search_hybrid_rrf_done",
                    mode=mode,
                    requested_limit=limit,
                    image_candidates=len(image_results),
                    text_candidates=len(text_results),
                    fused_candidates=len(raw_results),
                    image_search_ms=image_search_ms,
                    text_search_ms=text_search_ms,
                    fusion_ms=fusion_ms,
                    index_version=self.index_manager.active_version,
                )
            else:
                log.warning(
                    "search_hybrid_text_index_unavailable",
                    mode=mode,
                    requested_limit=limit,
                    index_version=self.index_manager.active_version,
                )
                raw_results = image_results
                log.info(
                    "search_hybrid_fallback_image_only_done",
                    mode=mode,
                    requested_limit=limit,
                    image_candidates=len(image_results),
                    image_search_ms=image_search_ms,
                    index_version=self.index_manager.active_version,
                )

        if not raw_results:
            log.info(
                "search_results_empty",
                mode=mode,
                requested_limit=limit,
                index_version=self.index_manager.active_version,
            )
            return []

        hydrate_started = perf_counter()
        results = self._hydrate_results(raw_results, db)
        log.info(
            "search_hydration_done",
            mode=mode,
            requested_limit=limit,
            hydrated_count=len(results),
            hydration_ms=int((perf_counter() - hydrate_started) * 1000),
            index_version=self.index_manager.active_version,
        )
        return results

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
        self,
        raw_results: list[tuple[int, float]],
        db: Session,
    ) -> list[SearchResult]:
        image_ids = [result[0] for result in raw_results]

        image_rows = db.execute(
            select(ORMImage, Annotation)
            .outerjoin(Annotation, Annotation.image_id == ORMImage.id)
            .where(ORMImage.id.in_(image_ids))
        ).all()

        image_map = {image.id: (image, annotation) for image, annotation in image_rows}
        results: list[SearchResult] = []

        for image_id, score in raw_results:
            row = image_map.get(image_id)
            if row is None:
                continue

            image, annotation = row
            url = None
            if image.s3_key:
                url = self._storage.generate_presigned_url(
                    image.s3_key,
                    expiration=settings.s3_presigned_url_expiry,
                )

            results.append(
                SearchResult(
                    id=image.id,
                    sha256=image.sha256,
                    score=score,
                    url=url,
                    caption=annotation.caption_text if annotation else None,
                    ocr_text=annotation.ocr_text if annotation else None,
                    width=image.width,
                    height=image.height,
                )
            )

        return results


class SearchIndexExecution:
    def __init__(
        self,
        index_manager: FaissIndexManager,
        *,
        storage: StorageService | None = None,
        text_encoder_factory=SearchTextEncoder.get_instance,
    ) -> None:
        self._index_manager = index_manager
        self._storage = storage
        self._text_encoder_factory = text_encoder_factory

    def search(
        self,
        *,
        query: str,
        limit: int,
        offset: int,
        mode: Literal["image", "hybrid"],
    ) -> SearchIndexPage:
        start_time = time.perf_counter()
        self.ensure_index_loaded_for_thread()
        embedding = self._encode_query(query, mode)

        try:
            with read_session_scope() as db:
                service = SearchService(self._index_manager, storage=self._storage)
                results = service.search_by_embedding(
                    embedding=embedding,
                    limit=limit + offset,
                    db=db,
                    mode=mode,
                )
        except ValueError as exc:
            raise SearchInvalidRequestError(str(exc)) from exc
        except Exception as exc:
            log.exception("search_query_failed", query=query, mode=mode, limit=limit, offset=offset)
            raise SearchExecutionFailedError("Search failed") from exc

        paginated = results[offset : offset + limit]
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        log.info(
            "search_request_done",
            query=query,
            resolved_mode=mode,
            total_results=len(results),
            returned_results=len(paginated),
            limit=limit,
            offset=offset,
            search_time_ms=round(elapsed_ms, 2),
            index_version=self._index_manager.active_version,
        )

        return SearchIndexPage(
            query=query,
            results=paginated,
            total=len(results),
            limit=limit,
            offset=offset,
            search_time_ms=round(elapsed_ms, 2),
            index_version=self._index_manager.active_version,
        )

    def find_similar(self, *, image_id: int, limit: int) -> SearchIndexPage:
        start_time = time.perf_counter()
        self.ensure_index_loaded_for_thread()

        try:
            with read_session_scope() as db:
                service = SearchService(self._index_manager, storage=self._storage)
                results = service.find_similar(image_id=image_id, limit=limit, db=db)
        except ValueError as exc:
            raise SearchImageNotFoundError(str(exc)) from exc
        except Exception as exc:
            log.exception("similar_search_failed", image_id=image_id, limit=limit)
            raise SearchExecutionFailedError("Search failed") from exc

        results = [result for result in results if result.id != image_id][:limit]
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return SearchIndexPage(
            query=f"similar_to:{image_id}",
            results=results,
            total=len(results),
            limit=limit,
            offset=0,
            search_time_ms=round(elapsed_ms, 2),
            index_version=self._index_manager.active_version,
        )

    def ensure_index_loaded_for_thread(self) -> None:
        global _last_index_check
        now = time.monotonic()
        if self._index_manager.is_loaded and (now - _last_index_check) < _INDEX_CHECK_INTERVAL:
            return
        with read_session_scope() as db:
            self._ensure_index_loaded(db)
        _last_index_check = now

    def _ensure_index_loaded(self, db: Session) -> None:
        global _active_embed_model
        active_build = db.query(IndexBuild).filter(IndexBuild.is_active).first()
        if active_build is None:
            raise SearchIndexUnavailableError("Search index not loaded")
        _active_embed_model = active_build.embed_model

        if (
            not self._index_manager.is_loaded
            or self._index_manager.active_version != active_build.version
        ):
            try:
                self._index_manager.load_active_index(db)
            except FileNotFoundError as exc:
                raise SearchIndexUnavailableError("Search index not loaded") from exc
            except Exception as exc:
                log.exception("search_index_reload_failed")
                raise SearchIndexLoadError(f"Failed to load search index: {exc}") from exc

    def _encode_query(self, query: str, mode: str | None) -> list[float]:
        try:
            encoder = self._text_encoder_factory()
        except Exception as exc:
            log.exception("search_text_encoder_load_failed", query=query, mode=mode)
            raise SearchQueryEncodingError(f"Failed to load text encoder: {exc}") from exc

        check_encoder_index_compatibility(encoder, _active_embed_model)

        try:
            embedding_arr = encoder.encode(query)
            return embedding_arr.tolist()
        except Exception as exc:
            log.exception("search_query_encoding_failed", query=query, mode=mode)
            raise SearchQueryEncodingError(f"Failed to encode query: {exc}") from exc
