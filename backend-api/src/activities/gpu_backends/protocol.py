from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from activities.embedding.models import (
        EmbedBatchInput,
        EmbedBatchOutput,
        EncodeQueryInput,
        EncodeQueryOutput,
    )
    from activities.vision.models import CaptionInput, CaptionOutput, OCRInput, OCROutput


@runtime_checkable
class GpuBackend(Protocol):
    async def caption(self, input: CaptionInput) -> CaptionOutput: ...
    async def ocr(self, input: OCRInput) -> OCROutput: ...
    async def embed_batch(self, input: EmbedBatchInput) -> EmbedBatchOutput: ...
    async def encode_query(self, input: EncodeQueryInput) -> EncodeQueryOutput: ...
