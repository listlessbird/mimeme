from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import modal

from activities.embedding.models import (
    EmbedBatchInput,
    EmbedBatchOutput,
    EmbedImageOutput,
    EncodeQueryInput,
    EncodeQueryOutput,
)
from activities.vision.models import CaptionInput, CaptionOutput, OCRInput, OCROutput

if TYPE_CHECKING:
    from modal_app import EmbeddingService, VisionService

MODAL_APP_NAME = "findmeme-gpu"


class ModalGpuBackend:
    _vision_cls: type[VisionService]
    _embedding_cls: type[EmbeddingService]

    def __init__(self) -> None:
        self._vision_cls = modal.Cls.from_name(MODAL_APP_NAME, "VisionService")  # type: ignore[assignment]
        self._embedding_cls = modal.Cls.from_name(MODAL_APP_NAME, "EmbeddingService")  # type: ignore[assignment]

    async def caption(self, input: CaptionInput) -> CaptionOutput:
        vision = self._vision_cls()

        result = await asyncio.to_thread(
            vision.caption.remote, s3_key=input.s3_key, length=input.length
        )

        return CaptionOutput(
            image_id=input.image_id, caption=result["caption"], model=result["model"]
        )

    async def ocr(self, input: OCRInput) -> OCROutput:
        vision = self._vision_cls()

        result = await asyncio.to_thread(vision.ocr.remote, s3_key=input.s3_key)

        return OCROutput(image_id=input.image_id, text=result["text"], model=result["model"])

    async def embed_batch(self, input: EmbedBatchInput) -> EmbedBatchOutput:
        items = [
            {
                "image_id": item.image_id,
                "s3_key": item.s3_key,
                "text": item.text,
                "sha256": item.sha256,
                "dataset": item.dataset or input.dataset,
            }
            for item in input.items
        ]

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

    async def encode_query(self, input: EncodeQueryInput) -> EncodeQueryOutput:
        embedding = self._embedding_cls()
        result = await asyncio.to_thread(embedding.encode_text.remote, query=input.query)
        return EncodeQueryOutput(
            embedding=result["embedding"],
            model=result["model"],
            dimension=result["dimension"],
        )
