from __future__ import annotations

import pytest
from pydantic import ValidationError
from temporalio.contrib.pydantic import pydantic_data_converter

from mimeme import inference
from mimeme.inference.model import (
    Annotation,
    Batch,
    BatchResult,
    Embedding,
    Failed,
    Input,
    Item,
    Ok,
    image_embedding_key,
    text_embedding_key,
)


async def _roundtrip(value: object) -> object:
    payloads = await pydantic_data_converter.encode([value])
    (decoded,) = await pydantic_data_converter.decode(payloads, [type(value)])
    return decoded


async def test_input_roundtrip_through_temporal_converter() -> None:
    value = Input(image_id=7, media_key="images/a.jpg", length="long")
    assert await _roundtrip(value) == value


async def test_batch_result_roundtrip_and_projections() -> None:
    result = BatchResult(
        items=[
            Ok(
                embedding=Embedding(
                    image_id=1,
                    image_embedding_key="e/1.npy",
                    text_embedding_key="e/1_text.npy",
                    model="m",
                    dimension=768,
                )
            ),
            Failed(image_id=2, error="boom"),
        ]
    )
    decoded = await _roundtrip(result)
    assert decoded == result
    assert [e.image_id for e in result.results] == [1]
    assert result.failed_ids == [2]


def test_contracts_are_frozen() -> None:
    ann = Annotation(image_id=1, caption="c", caption_model="m", ocr_text="", ocr_model="m")
    with pytest.raises(ValidationError):
        ann.caption = "x"  # type: ignore[misc]


def test_unknown_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        Item.model_validate({"image_id": 1, "media_key": "k", "sha256": "s", "surprise": True})


def test_batch_item_discriminated_union() -> None:
    batch = Batch(items=[Item(image_id=1, media_key="k", sha256="s", text="t")])
    assert batch.items[0].dataset is None


def test_embedding_key_rules() -> None:
    key = image_embedding_key(sha256="abc", model="google/siglip2", dataset=None)
    assert key == "embeddings/google_siglip2/api-ingested/abc.npy"
    assert text_embedding_key(key) == "embeddings/google_siglip2/api-ingested/abc_text.npy"


def test_module_exports_client_interface() -> None:
    assert hasattr(inference, "Client")
    assert hasattr(inference, "create")
