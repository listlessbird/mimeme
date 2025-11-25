from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterator, Tuple

from ingestion.embeddings.db_ops import (
    ensure_artifact_dir,
    get_pending_embeddings,
    get_image_text,
    mark_embedding_status,
    save_embeddings,
)

from .base import EmbedderConfig
from .siglip_embedder import SiglipEmbedder
from ..config import CFG


def run_embedding_loop(
    cfg: EmbedderConfig,
    limit: Optional[int] = None,
) -> Iterator[Tuple[int, int]]:

    embedder = SiglipEmbedder(cfg)
    model_name = embedder.image_model_name
    artifact_dir = ensure_artifact_dir(model_name)

    rows = get_pending_embeddings(limit)

    if not rows:
        print("No pending images to embed")
        return

    total = len(rows)
    yield 0, total

    for i in range(0, total, cfg.batch_size):
        batch = rows[i: i + cfg.batch_size]
        batch_ids = [img_id for img_id, _ in batch]

        try:
            mark_embedding_status(batch_ids, "running")

            image_paths = [
                (img_id, str(Path(CFG.image_root) / rel_path))
                for img_id, rel_path in batch
            ]

            results = embedder.embed_batch(
                image_paths=image_paths,
                text_provider=get_image_text
            )

            for img_id, img_emb, txt_emb in results:
                try:
                    save_embeddings(
                        image_id=img_id,
                        image_embedding=img_emb,
                        text_embedding=txt_emb,
                        model_name=model_name,
                        artifact_dir=artifact_dir
                    )
                except Exception as e:
                    print(f"Failed to save embeddings for image {img_id}: {e}")
                    mark_embedding_status([img_id], "failed")
            
            yield len(batch), total

        except Exception as e:
            print(f"Embedding batch failed: {e}")
            mark_embedding_status(batch_ids, "failed")


def create_embedder(cfg: EmbedderConfig) -> SiglipEmbedder:
    return SiglipEmbedder(cfg)
