from pathlib import Path
from typing import Any

import numpy as np
import structlog

from api.config import settings
from api.deps import get_index_manager
from api.tasks import celery_app
from ingestion.orm import Artifact, Processing, session_scope

log = structlog.get_logger()


@celery_app.task(
    bind=True, name="api.tasks.index.rebuild_index", max_retries=2, default_retry_delay=300
)
def rebuild_index_task(self, force: bool = False, model_name: str | None = None) -> dict[str, Any]:
    model_name = model_name or settings.embed_model

    index_manager = get_index_manager()

    self.update_state(
        state="PROGRESS",
        meta={"progress": 0, "message": "Collecting embeddings..."},
    )

    with session_scope() as session:
        artifacts = (
            session.query(Artifact)
            .filter(
                Artifact.kind == "embed", Artifact.model_version.contains(model_name.split("/")[-1])
            )
            .all()
        )

        done_ids = set(
            row[0]
            for row in session.query(Processing.image_id)
            .filter(Processing.embed_status == "done")
            .all()
        )

        valid_artifacts = [a for a in artifacts if a.image_id in done_ids]

    if not valid_artifacts:
        log.warning("no_embeddings_found")
        return {
            "status": "skipped",
            "reason": "No embeddings found",
            "num_vectors": 0,
        }

    log.info("found_embeddings", count=len(valid_artifacts))

    self.update_state(
        state="PROGRESS",
        meta={
            "progress": 10,
            "message": f"Loading {len(valid_artifacts)} embeddings...",
        },
    )

    embeddings = []
    image_ids = []
    failed_loads = 0

    artifact_base = Path(settings.db_path).parent

    for i, artifact in enumerate(valid_artifacts):
        try:
            embed_path = artifact_base / artifact.path if artifact.path else None

            if embed_path is not None:
                if not embed_path.exists():
                    alt_path = (
                        settings.artifact_dir / artifact.model_version / f"{artifact.image_id}.npy"
                    )

                    if not alt_path.exists():
                        failed_loads += 1
                        continue
                    embed_path = alt_path

                embedding = np.load(embed_path)
                embeddings.append(embedding)
                image_ids.append(artifact.image_id)

                if i % 100 == 0:
                    progress = 10 + (i / len(valid_artifacts)) * 40
                    self.update_state(
                        state="PROGRESS",
                        meta={
                            "progress": progress,
                            "message": f"Loaded {i}/{len(valid_artifacts)} embeddings",
                        },
                    )

        except Exception:
            log.error("failed_to_load_embedding", artifact_id=artifact.image_id)
            failed_loads += 1

        if not embeddings:
            return {
                "status": "failed",
                "reason": "No embeddings could be loaded",
                "failed_loads": failed_loads,
            }

        log.info(
            "embeddings_loaded",
            count=len(embeddings),
            failed=failed_loads,
        )

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
        )
        self.update_state(
            state="PROGRESS",
            meta={"progress": 80, "message": "Validating index..."},
        )
        # todo
        self.update_state(
            state="PROGRESS",
            meta={"progress": 90, "message": "Swapping to new index..."},
        )

        index_manager.swap_to_version(version)

        removed = index_manager.garbage_collect()

        log.info(
            "index_rebuild_complete",
            version=version,
            num_vectors=len(image_ids),
            dimension=embedding_matrix.shape[1],
            removed_versions=removed,
        )

        return {
            "status": "success",
            "version": version,
            "num_vectors": len(image_ids),
            "dimension": int(embedding_matrix.shape[1]),
            "failed_loads": failed_loads,
            "removed_versions": removed,
        }


@celery_app.task(name="api.tasks.index.cleanup_indexes")
def cleanup_indexes_task() -> dict[str, Any]:
    index_manager = get_index_manager()
    removed = index_manager.garbage_collect()

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
