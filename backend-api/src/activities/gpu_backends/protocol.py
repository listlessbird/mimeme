from __future__ import annotations

from typing import Protocol, runtime_checkable

from activities.embedding import (
    EmbedBatchInput,
    EmbedBatchOutput,
    EncodeQueryInput,
    EncodeQueryOutput,
)
from activities.vision import CaptionInput, CaptionOutput, OCRInput, OCROutput


@runtime_checkable
class GpuBackend(Protocol):
    async def caption(self, input: CaptionInput) -> CaptionOutput: ...
    async def ocr(self, input: OCRInput) -> OCROutput: ...
    async def embed_batch(self, input: EmbedBatchInput) -> EmbedBatchOutput: ...
    async def encode_query(self, input: EncodeQueryInput) -> EncodeQueryOutput: ...
