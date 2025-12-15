from typing import Any

import numpy as np
import structlog
from celery import Task

from api.config import settings
from api.deps import get_index_manager
from api.models.orm import Processing, session_scope
from api.services.storage import get_storage_service
from api.tasks import celery_app

log = structlog.get_logger()


@celery_app.task(
    bind=True, name="api.tasks.index.rebuild_index", max_retries=2, default_retry_delay=300
)
def rebuild_index_task(
    self: Task, force: bool = False, model_name: str | None = None
) -> dict[str, Any]:
    model_name = model_name or settings.embed_model
    storage = get_storage_service()
    index_manager = get_index_manager()

    self.update_state(
        state="PROGRESS",
        meta={"progress": 0, "message": "Collecting embeddings..."},
    )

    with session_scope() as session:
        done_procs = (
            session.query(Processing)
            .filter(Processing.embed_status == "done", Processing.embed_s3_key.isnot(None))
            .all()
        )

        if not done_procs:
            log.warning("no_embeddings_found", model_name=model_name)
            return {
                "status": "skipped",
                "reason": "no embeddings found",
                "num_vectors": 0,
            }

        log.info("found_embeddings", count=len(done_procs))

        self.update_state(
            state="PROGRESS",
            meta={"progress": 10, "message": "Loading embeddings..."},
        )

        embeddings = []
        image_ids = []
        failed_loads = 0

        for i, proc in enumerate(done_procs):
            try:
                if proc.embed_s3_key and storage.exists(proc.embed_s3_key):
                    embedding = storage.download_numpy(proc.embed_s3_key)
                    embeddings.append(embedding)
                    image_ids.append(proc.image_id)
                else:
                    failed_loads += 1
                    log.warning(
                        "embeddig_not_found_in_storage",
                        image_id=proc.image_id,
                        key=proc.embed_s3_key,
                    )

                if i % 100 == 0:
                    progress = 10 + (i / len(done_procs)) * 40
                    self.update_state(
                        state="PROGRESS",
                        meta={
                            "progress": progress,
                            "message": f"Downloaded {i} / {len(done_procs)} embeddings...",
                        },
                    )

            except Exception as e:
                log.error("failed_to_load_embeddings", image_id=proc.image_id, error=str(e))
                failed_loads += 1

        if not embeddings:
            return {
                "status": "failed",
                "reason": "failed to load any embeddings",
                "failed_loads": failed_loads,
            }

        log.info("loaded_embeddings", count=len(embeddings), failed_loads=failed_loads)

        self.update_state(
            state="PROGRESS",
            meta={"progress": 50, "message": "Building index..."},
        )

        embedding_matrix = np.stack(embeddings, axis=0)

        version = index_manager.build_index(
            embeddings=embedding_matrix,
            image_ids=image_ids,
            model_name=model_name,
            index_type=settings.index_type,
            db=session,
        )

        self.update_state(
            state="PROGRESS",
            meta={"progress": 80, "message": "Validating index..."},
        )

        self.update_state(
            state="PROGRESS",
            meta={"progress": 90, "message": "Swapping index..."},
        )

        index_manager.swap_to_version(version, session)
        removed = index_manager.garbage_collect(session)

        log.info(
            "rebuild_index_completed",
            version=version,
            num_vectors=len(embedding_matrix),
            removed_indexes=removed,
            dimension=embedding_matrix.shape[1],
        )

        return {
            "status": "success",
            "version": version,
            "num_vectors": len(embedding_matrix),
            "dimension": embedding_matrix.shape[1],
            "removed_indexes": removed,
            "failed_loads": failed_loads,
        }


@celery_app.task(name="api.tasks.index.cleanup_indexes")
def cleanup_indexes_task() -> dict[str, Any]:
    index_manager = get_index_manager()

    with session_scope() as session:
        removed = index_manager.garbage_collect(session)

    return {
        "removed": removed,
        "count": len(removed),
    }


@celery_app.task(
    bind=True,
    name="api.tasks.index.incremental_update",
)
def incremental_update_task(
    self,
    image_ids: list[int],
) -> dict[str, Any]:
    # todo
    #  implement a proper delta index strateg
    return rebuild_index_task.apply_async(kwargs={"force": True}).get()
