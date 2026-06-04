from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from activities.embedding.models import (
        EmbedBatchInput,
        EmbedBatchOutput,
    )
    from activities.vision.models import AnnotateImageInput, AnnotateImageOutput


@runtime_checkable
class GpuBackend(Protocol):
    async def annotate_image(self, input: AnnotateImageInput) -> AnnotateImageOutput: ...
    async def embed_batch(self, input: EmbedBatchInput) -> EmbedBatchOutput: ...
