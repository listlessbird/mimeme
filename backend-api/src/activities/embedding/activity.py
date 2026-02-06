from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import cast

import structlog
from PIL import Image as PILImage
from temporalio import activity
from temporalio.exceptions import ApplicationError

from activities.embedding.models import (
    EmbedBatchInput,
    EmbedBatchOutput,
    EmbedImageOutput,
    EncodeQueryInput,
    EncodeQueryOutput,
)
from activities.embedding.siglip import SiglipEmbedder
from shared.db import session_scope
from shared.models import ORMImage
from shared.services import get_storage_service
from shared.services.storage import StorageService


@activity.defn
async def embed_batch_activity(input: EmbedBatchInput) -> EmbedBatchOutput:
    started = time.monotonic()
    log = structlog.get_logger().bind(
        activity_name="embed_batch_activity",
        item_count=len(input.items),
        dataset=input.dataset,
    )
    outcome = "success"
    error_type: str | None = None
    error_message: str | None = None

    storage = cast(StorageService, get_storage_service())
    log.info("activity_step", step="start")
    try:
        embedder = SiglipEmbedder.get_instance()
    except (ImportError, OSError, RuntimeError) as exc:
        raise ApplicationError(
            "Embedding model dependencies are missing or incompatible.",
            non_retryable=True,
        ) from exc

    results: list[EmbedImageOutput] = []
    failed_ids: list[int] = []

    try:
        for item in input.items:
            item_log = log.bind(image_id=item.image_id, s3_key=item.s3_key)
            item_log.info("activity_step", step="item_start")
            try:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
                    tmp_path = Path(tmp.name)
                    item_log.info("activity_step", step="download_start")
                    storage.download_file(item.s3_key, tmp_path)
                    item_log.info("activity_step", step="download_complete", tmp_path=str(tmp_path))

                    pil_image = PILImage.open(tmp_path)

                    if pil_image.mode == "P" and "transparency" in pil_image.info:
                        pil_image = pil_image.convert("RGBA")
                    pil_image = pil_image.convert("RGB")

                    item_log.info("activity_step", step="encode_start")
                    img_feats = embedder.encode_images([pil_image])

                    txt_feats = embedder.encode_texts([item.text])
                    item_log.info("activity_step", step="encode_complete", dimension=int(img_feats.shape[-1]))

                    with session_scope() as session:
                        img = session.query(ORMImage).filter_by(id=item.image_id).first()

                        if not img:
                            raise ValueError(f"Image with id {item.image_id} not found in DB")

                        sha256 = img.sha256 if img else str(item.image_id)

                        dataset = img.dataset if img else input.dataset

                    img_embed_key = storage.build_embedding_key(
                        sha256=sha256,
                        model_name=embedder.image_model_name,
                        dataset=dataset,
                    )

                    text_embed_key = img_embed_key.replace(".npy", "_text.npy")

                    # todo: upload both embeddings in a single call concurrently
                    item_log.info("activity_step", step="upload_start")
                    storage.upload_numpy(img_feats[0], img_embed_key)
                    storage.upload_numpy(txt_feats[0], text_embed_key)
                    item_log.info("activity_step", step="upload_complete")

                    results.append(
                        EmbedImageOutput(
                            image_id=item.image_id,
                            image_embedding_key=img_embed_key,
                            text_embedding_key=text_embed_key,
                            model=embedder.image_model_name,
                            dimension=int(img_feats.shape[-1]),
                        )
                    )
                    item_log.info("activity_step", step="item_complete")
            except Exception as exc:
                failed_ids.append(item.image_id)
                item_log.error(
                    "activity_step",
                    step="item_failed",
                    error_type=type(exc).__name__,
                    error=str(exc),
                    exc_info=True,
                )
    except Exception as exc:
        outcome = "error"
        error_type = type(exc).__name__
        error_message = str(exc)
        raise
    finally:
        if failed_ids:
            outcome = "error"
        log.info(
            "activity_wide_event",
            event_type="activity_wide_event",
            outcome=outcome,
            duration_ms=int((time.monotonic() - started) * 1000),
            processed=len(results),
            failed=len(failed_ids),
            model=embedder.image_model_name if "embedder" in locals() else None,
            error_type=error_type,
            error=error_message,
        )

    return EmbedBatchOutput(results=results, failed_ids=failed_ids)


@activity.defn
async def encode_query_activity(input: EncodeQueryInput) -> EncodeQueryOutput:
    log = structlog.get_logger().bind(activity_name="encode_query_activity")
    log.info("activity_step", step="start")
    embedder = SiglipEmbedder.get_instance()
    query_embedding = embedder.encode_texts([input.query])[0]
    log.info("activity_wide_event", event_type="activity_wide_event", outcome="success", dimension=len(query_embedding))

    return EncodeQueryOutput(
        embedding=query_embedding.tolist(),
        model=embedder.image_model_name,
        dimension=len(query_embedding),
    )
