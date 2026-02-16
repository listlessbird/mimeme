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
    CollectEmbeddingsOutput,
    GarbageCollectOutput,
    SwapIndexInput,
)
from shared.db import session_scope
from shared.models import Processing, ProcessingStatus
from shared.services.storage import get_storage_service

log = structlog.get_logger()
MAX_DOWNLOAD_WORKERS = max(4, min(32, (os.cpu_count() or 8) * 2))


def _activity_context() -> dict[str, object]:
    try:
        info = activity.info()
    except RuntimeError:
        return {}
    return {
        "workflow_id": info.workflow_id,
        "run_id": info.workflow_run_id,
        "workflow_type": info.workflow_type,
        "activity_id": info.activity_id,
        "activity_type": info.activity_type,
        "attempt": info.attempt,
        "task_queue": info.task_queue,
        "is_local": info.is_local,
    }


def _emit_activity_event(
    *,
    activity_name: str,
    started_at: float,
    outcome: str,
    error: str | None = None,
    **fields: object,
) -> None:
    event: dict[str, object] = {
        "event_type": "activity_wide_event",
        "activity_name": activity_name,
        "outcome": outcome,
        "duration_ms": int((time.monotonic() - started_at) * 1000),
        **_activity_context(),
    }
    event.update(fields)
    if error:
        event["error"] = error
    log.info("activity_wide_event", **event)


@activity.defn
def collect_embeddings_activity() -> CollectEmbeddingsOutput:
    started = time.monotonic()
    outcome = "success"
    error_message: str | None = None
    total: int | None = None
    try:
        with session_scope() as session:
            done_procs = (
                session.query(Processing)
                .filter(
                    Processing.embed_status == ProcessingStatus.DONE,
                    Processing.embed_s3_key.isnot(None),
                )
                .all()
            )

            embedding_keys = [p.embed_s3_key for p in done_procs if p.embed_s3_key]
            image_ids = [p.image_id for p in done_procs]
            total = len(done_procs)

            return CollectEmbeddingsOutput(
                embedding_keys=embedding_keys,
                image_ids=image_ids,
                total=total,
            )
    except Exception as exc:
        outcome = "error"
        error_message = str(exc)
        raise
    finally:
        _emit_activity_event(
            activity_name="collect_embeddings_activity",
            started_at=started,
            outcome=outcome,
            error=error_message,
            total=total,
        )


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
                (image_id, embed_s3_key)
                for image_id, embed_s3_key in done_procs
                if embed_s3_key
            ]

            if not done_procs:
                raise ValueError("No embeddings found to build index")

            total_candidates = len(candidates)
            image_ids: list[int] = [0] * total_candidates
            embedding_matrix: np.ndarray | None = None
            loaded = 0

            log.info(
                "activity_step",
                activity_name="build_index_activity",
                step="start_collect_embeddings",
                total_candidates=total_candidates,
                model_name=input.model_name,
                index_type=input.index_type,
            )

            def _download_one(candidate: tuple[int, str]) -> tuple[int, np.ndarray] | None:
                image_id, embed_s3_key = candidate
                try:
                    return image_id, storage.download_numpy(embed_s3_key)
                except ClientError:
                    return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
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
                                "missing": missing_vectors,
                                "workers": MAX_DOWNLOAD_WORKERS,
                            }
                        )

                    result = future.result()
                    if result is None:
                        missing_vectors += 1
                        continue
                    image_id, embedding = result
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

            if loaded == 0 or embedding_matrix is None:
                raise ValueError("Failed to load any embeddings from storage")

            if loaded < total_candidates:
                embedding_matrix = embedding_matrix[:loaded]
                image_ids = image_ids[:loaded]
            num_vectors = loaded
            dimension = int(embedding_matrix.shape[1])

            log.info(
                "activity_step",
                activity_name="build_index_activity",
                step="embeddings_collected",
                loaded=loaded,
                missing=missing_vectors,
                total_candidates=total_candidates,
            )

            activity.heartbeat(
                {
                    "stage": "build_faiss_index",
                    "loaded": loaded,
                    "dimension": dimension,
                }
            )
            version = index_manager.build_index(
                embeddings=embedding_matrix,
                image_ids=image_ids,
                model_name=input.model_name,
                index_type=input.index_type,
                db=session,
            )

            index_key = storage.build_index_key(version, "index.faiss")
            return BuildIndexOutput(
                version=version,
                num_vectors=num_vectors,
                dimension=dimension,
                s3_key=index_key,
            )
    except Exception as exc:
        outcome = "error"
        error_message = str(exc)
        raise
    finally:
        _emit_activity_event(
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
        _emit_activity_event(
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
        _emit_activity_event(
            activity_name="garbage_collect_indexes_activity",
            started_at=started,
            outcome=outcome,
            error=error_message,
            removed_versions=removed_count,
        )
