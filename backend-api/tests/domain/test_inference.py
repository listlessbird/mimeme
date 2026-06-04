from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from activities.embedding.models import EmbedBatchInput, EmbedImageInput
from activities.vision.moondream import Moondream2
from domain.inference import (
    build_image_embedding_key,
    build_text_embedding_key_for_image_embedding,
    pooled_features_to_numpy,
    prepare_rgb_image_for_inference,
    select_pooled_feature_tensor,
    to_modal_embedding_item,
)


def test_prepare_rgb_image_for_inference_converts_rgb() -> None:
    image = Image.new("L", (2, 2))

    result = prepare_rgb_image_for_inference(image)

    assert result.mode == "RGB"


def test_prepare_rgb_image_for_inference_handles_palette_transparency() -> None:
    image = Image.new("P", (2, 2))
    image.info["transparency"] = 0

    result = prepare_rgb_image_for_inference(image)

    assert result.mode == "RGB"


def test_moondream_annotation_reuses_encoded_image() -> None:
    backend = object.__new__(Moondream2)
    backend.config = SimpleNamespace(model_id="model", revision="rev")

    class Model:
        encode_calls = 0
        caption_image: object | None = None
        query_image: object | None = None
        query_reasoning: bool | None = None

        def encode_image(self, image: Image.Image) -> str:
            self.encode_calls += 1
            return "encoded-image"

        def caption(self, image: object, length: str = "normal") -> dict[str, str]:
            self.caption_image = image
            return {"caption": f"{length} caption"}

        def query(self, image: object, question: str, reasoning: bool = True) -> dict[str, str]:
            self.query_image = image
            self.query_reasoning = reasoning
            return {"answer": question}

    model = Model()
    backend.model = model

    result = backend.annotate_image(Image.new("RGB", (1, 1)), length="long")

    assert model.encode_calls == 1
    assert model.caption_image == "encoded-image"
    assert model.query_image == "encoded-image"
    assert model.query_reasoning is False
    assert result.caption == "long caption"
    assert result.caption_model == "model@rev"
    assert result.ocr_text == "Transcribe the text in natural reading order."
    assert result.ocr_model == "model@rev"


def test_select_pooled_feature_tensor_from_direct_tensor() -> None:
    tensor = torch.tensor([[1.0, 2.0]])

    assert select_pooled_feature_tensor(tensor, kind="image") is tensor


def test_select_pooled_feature_tensor_from_named_fields() -> None:
    image_embeds = torch.tensor([[1.0, 2.0]])
    text_embeds = torch.tensor([[3.0, 4.0]])
    pooler_output = torch.tensor([[5.0, 6.0]])

    assert torch.equal(
        select_pooled_feature_tensor(SimpleNamespace(image_embeds=image_embeds), kind="image"),
        image_embeds,
    )
    assert torch.equal(
        select_pooled_feature_tensor(SimpleNamespace(text_embeds=text_embeds), kind="text"),
        text_embeds,
    )
    assert torch.equal(
        select_pooled_feature_tensor(SimpleNamespace(pooler_output=pooler_output), kind="text"),
        pooler_output,
    )


def test_unknown_tensor_output_raises() -> None:
    with pytest.raises(ValueError):
        select_pooled_feature_tensor(SimpleNamespace(), kind="text")


def test_pooled_features_to_numpy() -> None:
    output = SimpleNamespace(text_embeds=torch.tensor([[1.0, 2.0]]))

    result = pooled_features_to_numpy(output, kind="text")

    assert result.tolist() == [[1.0, 2.0]]


def test_embedding_key_generation_with_and_without_dataset() -> None:
    assert build_image_embedding_key(sha256="abc", model_name="google/siglip", dataset="memes") == (
        "embeddings/google_siglip/memes/abc.npy"
    )
    assert build_image_embedding_key(sha256="abc", model_name="google/siglip", dataset=None) == (
        "embeddings/google_siglip/api-ingested/abc.npy"
    )


def test_build_text_embedding_key_for_image_embedding() -> None:
    assert (
        build_text_embedding_key_for_image_embedding("embeddings/google_siglip/memes/abc.npy")
        == "embeddings/google_siglip/memes/abc_text.npy"
    )


def test_modal_embedding_payload_uses_item_dataset_before_batch_dataset() -> None:
    batch = EmbedBatchInput(
        items=[
            EmbedImageInput(
                image_id=1,
                s3_key="images/1.jpg",
                text="hello",
                sha256="abc",
                dataset="item-dataset",
            )
        ],
        dataset="batch-dataset",
    )

    payload = to_modal_embedding_item(batch.items[0], batch.dataset)

    assert payload == {
        "image_id": 1,
        "s3_key": "images/1.jpg",
        "text": "hello",
        "sha256": "abc",
        "dataset": "item-dataset",
    }
