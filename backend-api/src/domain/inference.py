from __future__ import annotations

from typing import Literal, Protocol, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from PIL.Image import Image


class EmbeddingPayloadItem(Protocol):
    image_id: int
    s3_key: str
    text: str
    sha256: str
    dataset: str | None


class ModalEmbeddingItemPayload(TypedDict):
    image_id: int
    s3_key: str
    text: str
    sha256: str
    dataset: str | None


class PooledFeatureTensor(Protocol):
    def detach(self) -> PooledFeatureTensor: ...

    def cpu(self) -> PooledFeatureTensor: ...

    def numpy(self) -> NDArray[np.float32]: ...


def prepare_rgb_image_for_inference(image: Image) -> Image:
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    return image.convert("RGB")


def build_image_embedding_key(*, sha256: str, model_name: str, dataset: str | None) -> str:
    return f"embeddings/{model_name.replace('/', '_')}/{dataset or 'api-ingested'}/{sha256}.npy"


def build_text_embedding_key_for_image_embedding(image_key: str) -> str:
    return image_key.replace(".npy", "_text.npy")


def to_modal_embedding_item(
    item: EmbeddingPayloadItem,
    batch_dataset: str | None,
) -> ModalEmbeddingItemPayload:
    return {
        "image_id": item.image_id,
        "s3_key": item.s3_key,
        "text": item.text,
        "sha256": item.sha256,
        "dataset": item.dataset or batch_dataset,
    }


def select_pooled_feature_tensor(
    output: object,
    *,
    kind: Literal["image", "text"],
) -> PooledFeatureTensor:
    if hasattr(output, "cpu"):
        return cast(PooledFeatureTensor, output)

    candidates = (
        ("image_embeds", "pooler_output") if kind == "image" else ("text_embeds", "pooler_output")
    )

    for field in candidates:
        if hasattr(output, field):
            value = getattr(output, field)
            if hasattr(value, "cpu"):
                return cast(PooledFeatureTensor, value)

    raise ValueError(f"Unknown {kind} model output format")


def pooled_features_to_numpy(
    output: object,
    *,
    kind: Literal["image", "text"],
) -> NDArray[np.float32]:
    tensor = select_pooled_feature_tensor(output, kind=kind)
    return tensor.cpu().numpy()
