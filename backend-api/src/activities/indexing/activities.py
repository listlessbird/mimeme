from __future__ import annotations

import concurrent.futures
import os
import time

import numpy as np
import structlog
from botocore.exceptions import ClientError
from temporalio import activity

from activities.indexing.faiss_manager import FaissIndexManager
from activities.indexing.models import (
    BuildIndexInput,
    BuildIndexOutput,
    GarbageCollectOutput,
    SwapIndexInput,
)
from shared.db import session_scope
from shared.logging import emit_activity_event
from shared.models import Processing, ProcessingStatus
from shared.services.storage import get_storage_service

log = structlog.get_logger()
MAX_DOWNLOAD_WORKERS = max(4, min(32, (os.cpu_count() or 8) * 2))


@activity.defn
def build_index_activity(input: BuildIndexInput) -> BuildIndexOutput:
    started = time.monotonic()
    outcome = "success"
    error_message: str | None = None
    total_candidates: int | None = None
    missing_vectors = 0
    num_vectors: int | None = None
    dimension: int | None = None
    version: str | None = None
    text_num_vectors: int | None = None
    storage = get_storage_service()
    index_manager = FaissIndexManager.get_instance()
    try:
        with session_scope() as session:
            done_procs = (
                session.query(Processing.image_id, Processing.embed_s3_key)
                .filter(
                    Processing.embed_status == ProcessingStatus.DONE,
                    Processing.embed_s3_key.isnot(None),
                )
                .all()
            )
            total_candidates = len(done_procs)
            candidates = [
                (image_id, embed_s3_key) for image_id, embed_s3_key in done_procs if embed_s3_key
            ]

            if not done_procs:
                raise ValueError("No embeddings found to build index")

            total_candidates = len(candidates)
            image_ids: list[int] = [0] * total_candidates
            text_image_ids: list[int] = [0] * total_candidates
            embedding_matrix: np.ndarray | None = None
            text_embedding_matrix: np.ndarray | None = None
            loaded = 0
            text_loaded = 0

            log.info(
                "activity_step",
                activity_name="build_index_activity",
                step="start_collect_embeddings",
                total_candidates=total_candidates,
                model_name=input.model_name,
                index_type=input.index_type,
            )

            def _download_one(
                candidate: tuple[int, str],
            ) -> tuple[int, np.ndarray, np.ndarray | None] | None:
                image_id, embed_s3_key = candidate
                try:
                    img_emb = storage.download_numpy(embed_s3_key)
                except ClientError:
                    return None
                # Try to download the companion text embedding
                text_key = embed_s3_key.replace(".npy", "_text.npy")
                txt_emb: np.ndarray | None = None
                try:
                    txt_emb = storage.download_numpy(text_key)
                except Exception:
                    pass
                return image_id, img_emb, txt_emb

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_DOWNLOAD_WORKERS
            ) as executor:
                futures = [
                    executor.submit(_download_one, (image_id, embed_s3_key))
                    for image_id, embed_s3_key in candidates
                ]
                for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                    if i == 1 or i % 100 == 0:
                        activity.heartbeat(
                            {
                                "stage": "collect_embeddings",
                                "scanned": i,
                                "total": total_candidates,
                                "loaded": loaded,
                                "text_loaded": text_loaded,
                                "missing": missing_vectors,
                                "workers": MAX_DOWNLOAD_WORKERS,
                            }
                        )

                    result = future.result()
                    if result is None:
                        missing_vectors += 1
                        continue
                    image_id, embedding, text_embedding = result
                    embedding = embedding.astype(np.float32, copy=False)

                    if embedding_matrix is None:
                        dimension = int(embedding.shape[0])
                        embedding_matrix = np.empty((total_candidates, dimension), dtype=np.float32)

                    if dimension is not None and int(embedding.shape[0]) != dimension:
                        raise ValueError(
                            f"Embedding dimension mismatch for image_id={image_id}: "
                            f"expected {dimension}, got {int(embedding.shape[0])}"
                        )

                    image_ids[loaded] = image_id
                    assert embedding_matrix is not None
                    embedding_matrix[loaded, :] = embedding
                    loaded += 1

                    # Collect text embeddings when available
                    if text_embedding is not None:
                        text_embedding = text_embedding.astype(np.float32, copy=False)
                        if text_embedding_matrix is None:
                            text_dim = int(text_embedding.shape[0])
                            text_embedding_matrix = np.empty(
                                (total_candidates, text_dim), dtype=np.float32
                            )
                        if int(text_embedding.shape[0]) != text_dim:
                            log.warning(
                                "activity_step",
                                activity_name="build_index_activity",
                                step="text_embedding_dim_mismatch",
                                image_id=image_id,
                                expected=text_dim,
                                got=int(text_embedding.shape[0]),
                            )
                        else:
                            text_image_ids[text_loaded] = image_id
                            text_embedding_matrix[text_loaded, :] = text_embedding
                            text_loaded += 1

            if loaded == 0 or embedding_matrix is None:
                raise ValueError("Failed to load any embeddings from storage")

            if loaded < total_candidates:
                embedding_matrix = embedding_matrix[:loaded]
                image_ids = image_ids[:loaded]
            num_vectors = loaded
            dimension = int(embedding_matrix.shape[1])

            # Trim text embedding matrix
            final_text_embeddings: np.ndarray | None = None
            final_text_image_ids: list[int] | None = None
            if text_loaded > 0 and text_embedding_matrix is not None:
                final_text_embeddings = text_embedding_matrix[:text_loaded]
                final_text_image_ids = text_image_ids[:text_loaded]

            log.info(
                "activity_step",
                activity_name="build_index_activity",
                step="embeddings_collected",
                loaded=loaded,
                text_loaded=text_loaded,
                missing=missing_vectors,
                total_candidates=total_candidates,
            )

            activity.heartbeat(
                {
                    "stage": "build_faiss_index",
                    "loaded": loaded,
                    "text_loaded": text_loaded,
                    "dimension": dimension,
                }
            )
            build_result = index_manager.build_index(
                embeddings=embedding_matrix,
                image_ids=image_ids,
                model_name=input.model_name,
                index_type=input.index_type,
                db=session,
                text_embeddings=final_text_embeddings,
                text_image_ids=final_text_image_ids,
            )

            version = build_result.version
            text_num_vectors = build_result.text_num_vectors

            index_key = storage.build_index_key(version, "index.faiss")
            return BuildIndexOutput(
                version=version,
                num_vectors=num_vectors,
                dimension=dimension,
                s3_key=index_key,
                text_num_vectors=text_num_vectors,
                text_s3_key=build_result.text_s3_key,
            )
    except Exception as exc:
        outcome = "error"
        error_message = str(exc)
        raise
    finally:
        emit_activity_event(
            log=log,
            activity_name="build_index_activity",
            started_at=started,
            outcome=outcome,
            error=error_message,
            model_name=input.model_name,
            index_type=input.index_type,
            force=input.force,
            total_candidates=total_candidates,
            missing_vectors=missing_vectors,
            version=version,
            num_vectors=num_vectors,
            dimension=dimension,
            text_num_vectors=text_num_vectors,
        )


@activity.defn
def swap_index_activity(input: SwapIndexInput) -> None:
    started = time.monotonic()
    outcome = "success"
    error_message: str | None = None
    index_manager = FaissIndexManager.get_instance()
    try:
        with session_scope() as session:
            index_manager.swap_to_version(input.version, session)
    except Exception as exc:
        outcome = "error"
        error_message = str(exc)
        raise
    finally:
        emit_activity_event(
            log=log,
            activity_name="swap_index_activity",
            started_at=started,
            outcome=outcome,
            error=error_message,
            version=input.version,
        )


@activity.defn
def garbage_collect_indexes_activity() -> GarbageCollectOutput:
    started = time.monotonic()
    outcome = "success"
    error_message: str | None = None
    removed_count: int | None = None
    index_manager = FaissIndexManager.get_instance()
    try:
        with session_scope() as session:
            removed = index_manager.garbage_collect(session)
            removed_count = len(removed)

        return GarbageCollectOutput(removed_versions=removed)
    except Exception as exc:
        outcome = "error"
        error_message = str(exc)
        raise
    finally:
        emit_activity_event(
            log=log,
            activity_name="garbage_collect_indexes_activity",
            started_at=started,
            outcome=outcome,
            error=error_message,
            removed_versions=removed_count,
        )
