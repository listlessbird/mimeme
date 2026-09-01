from __future__ import annotations

from typing import TYPE_CHECKING

from mimeme.inference import bge
from mimeme.inference.client import Client, Progress
from mimeme.inference.model import (
    CAPTION_PROMPT_VERSION,
    Annotation,
    Batch,
    BatchResult,
    Embedding,
    Error,
    Failed,
    Input,
    Invalid,
    Item,
    Ok,
    Timeout,
    Unavailable,
    embedding_prefix,
    image_embedding_key,
)

if TYPE_CHECKING:
    import httpx

    from mimeme.config import Settings


def create(settings: Settings, http: httpx.AsyncClient) -> Client:
    if settings.compute.gpu_backend == "modal":
        from mimeme.inference.modal import Modal

        return Modal(
            app_name=settings.compute.modal_app_name,
            embed_model=settings.inference.embed_model,
            poll_interval_s=settings.compute.poll_interval_s,
        )

    from mimeme.inference.local import Local

    inference_url = settings.compute.inference_gateway_url or settings.compute.gateway_url
    return Local(
        http,
        base_url=inference_url,
        embed_model=settings.inference.embed_model,
        poll_interval_s=settings.compute.poll_interval_s,
    )


__all__ = [
    "Annotation",
    "CAPTION_PROMPT_VERSION",
    "Batch",
    "BatchResult",
    "Client",
    "Embedding",
    "Error",
    "Failed",
    "Input",
    "Invalid",
    "Item",
    "Ok",
    "Progress",
    "Timeout",
    "Unavailable",
    "bge",
    "create",
    "embedding_prefix",
    "image_embedding_key",
]
