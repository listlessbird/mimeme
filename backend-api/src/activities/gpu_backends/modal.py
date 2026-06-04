from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import modal

from activities.embedding.models import (
    EmbedBatchInput,
    EmbedBatchOutput,
    EmbedImageOutput,
)
from activities.vision.models import AnnotateImageInput, AnnotateImageOutput
from domain.inference import to_modal_embedding_item
from shared.config import settings

if TYPE_CHECKING:
    from modal_app import EmbeddingService, VisionService


class ModalGpuBackend:
    _vision_cls: type[VisionService]
    _embedding_cls: type[EmbeddingService]

    def __init__(self) -> None:
        self._vision_cls = modal.Cls.from_name(settings.modal_app_name, "VisionService")  # type: ignore[assignment]
        self._embedding_cls = modal.Cls.from_name(settings.modal_app_name, "EmbeddingService")  # type: ignore[assignment]

    async def annotate_image(self, input: AnnotateImageInput) -> AnnotateImageOutput:
        vision = self._vision_cls()

        result = await asyncio.to_thread(
            vision.annotate_image.remote, s3_key=input.s3_key, length=input.length
        )

        return AnnotateImageOutput(
            image_id=input.image_id,
            caption=result["caption"],
            caption_model=result["caption_model"],
            ocr_text=result["ocr_text"],
            ocr_model=result["ocr_model"],
        )

    async def embed_batch(self, input: EmbedBatchInput) -> EmbedBatchOutput:
        items = [to_modal_embedding_item(item, input.dataset) for item in input.items]

        embedding = self._embedding_cls()
        result = await asyncio.to_thread(embedding.embed_batch.remote, items=items)

        return EmbedBatchOutput(
            results=[
                EmbedImageOutput(
                    image_id=r["image_id"],
                    image_embedding_key=r["image_embedding_key"],
                    text_embedding_key=r["text_embedding_key"],
                    model=r["model"],
                    dimension=r["dimension"],
                )
                for r in result["results"]
            ],
            failed_ids=result["failed_ids"],
        )
