from __future__ import annotations

from typing import TYPE_CHECKING

from mimeme.inference.client import Client, Progress
from mimeme.inference.model import (
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
    image_embedding_key,
    text_embedding_key,
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

    return Local(
        http,
        base_url=settings.compute.gateway_url,
        embed_model=settings.inference.embed_model,
        poll_interval_s=settings.compute.poll_interval_s,
    )


__all__ = [
    "Annotation",
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
    "create",
    "image_embedding_key",
    "text_embedding_key",
]
