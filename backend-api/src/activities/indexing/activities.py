from __future__ import annotations

import numpy as np
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


@activity.defn
async def collect_embeddings_activity() -> CollectEmbeddingsOutput:
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

        return CollectEmbeddingsOutput(
            embedding_keys=embedding_keys,
            image_ids=image_ids,
            total=len(done_procs),
        )


@activity.defn
async def build_index_activity(input: BuildIndexInput) -> BuildIndexOutput:
    storage = get_storage_service()
    index_manager = FaissIndexManager.get_instance()

    with session_scope() as session:
        done_procs = (
            session.query(Processing)
            .filter(
                Processing.embed_status == ProcessingStatus.DONE,
                Processing.embed_s3_key.isnot(None),
            )
            .all()
        )

        if not done_procs:
            raise ValueError("No embeddings found to build index")

        embeddings: list[np.ndarray] = []
        image_ids: list[int] = []

        for proc in done_procs:
            if proc.embed_s3_key and storage.exists(proc.embed_s3_key):
                embedding = storage.download_numpy(proc.embed_s3_key)
                embeddings.append(embedding)
                image_ids.append(proc.image_id)

        if not embeddings:
            raise ValueError("Failed to load any embeddings from storage")

        embedding_matrix = np.stack(embeddings, axis=0)

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
            num_vectors=len(embeddings),
            dimension=int(embedding_matrix.shape[1]),
            s3_key=index_key,
        )


@activity.defn
async def swap_index_activity(input: SwapIndexInput) -> None:
    index_manager = FaissIndexManager.get_instance()

    with session_scope() as session:
        index_manager.swap_to_version(input.version, session)


@activity.defn
async def garbage_collect_indexes_activity() -> GarbageCollectOutput:
    index_manager = FaissIndexManager.get_instance()

    with session_scope() as session:
        removed = index_manager.garbage_collect(session)

    return GarbageCollectOutput(removed_versions=removed)
