from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from activities.embedding.models import EmbedderConfig
from activities.embedding.siglip import SiglipEmbedder
from activities.vision.models import VisionModelConfig
from activities.vision.moondream import Moondream2
from api.services.text_encoder import SearchTextEncoder
from shared.config import settings

pytestmark = [
    pytest.mark.model_smoke,
    pytest.mark.skipif(
        os.environ.get("RUN_MODEL_SMOKE") != "1",
        reason="set RUN_MODEL_SMOKE=1 or run `just test-model`",
    ),
]


def _sample_image() -> Image.Image:
    image = Image.new("RGB", (640, 360), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((36, 42, 604, 318), outline="black", width=6)
    draw.ellipse((86, 92, 226, 232), fill="gold", outline="black", width=4)
    draw.rectangle((296, 118, 544, 246), fill="royalblue", outline="black", width=4)
    draw.text((312, 160), "MOONDREAM TEST", fill="white")
    return image


def _assert_embedding_contract(value: np.ndarray, expected_shape: tuple[int, ...]) -> None:
    assert value.shape == expected_shape
    assert np.issubdtype(value.dtype, np.floating)
    assert np.isfinite(value).all()


@pytest.fixture(autouse=True)
def release_model_singletons() -> Generator[None]:
    Moondream2.release_instance()
    SiglipEmbedder.release_instance()
    yield
    Moondream2.release_instance()
    SiglipEmbedder.release_instance()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_siglip_embedder_encodes_image_and_text() -> None:
    embedder = SiglipEmbedder(
        EmbedderConfig(
            image_model=settings.embed_model,
            device=settings.embed_device,
            use_bnb_4bit=False,
            fp16_fallback=True,
        )
    )

    image_features = embedder.encode_images([_sample_image()])
    text_features = embedder.encode_texts(["MOONDREAM TEST yellow circle blue rectangle"])

    _assert_embedding_contract(image_features, (1, 768))
    _assert_embedding_contract(text_features, (1, 768))


def test_search_text_encoder_matches_torch_reference() -> None:
    fixture = np.load(Path(__file__).parent / "fixtures" / "torch_text_reference.npz")
    queries = [str(query) for query in fixture["queries"]]
    reference_ids = fixture["input_ids"]
    reference_embeddings = fixture["embeddings"]

    encoder = SearchTextEncoder(
        repo_id=settings.onnx_text_encoder_repo,
        revision=settings.onnx_text_encoder_revision,
        variant=settings.onnx_text_encoder_variant,
        threads=settings.onnx_text_encoder_threads,
    )

    assert encoder.source_model == settings.embed_model

    for i, query in enumerate(queries):
        np.testing.assert_array_equal(
            encoder.tokenize(query)[0], reference_ids[i], err_msg=f"query: {query!r}"
        )

    embeddings = np.stack([encoder.encode(query) for query in queries])
    _assert_embedding_contract(embeddings, reference_embeddings.shape)
    assert embeddings.dtype == np.float32

    cosines = (embeddings * reference_embeddings).sum(axis=1) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(reference_embeddings, axis=1)
    )
    assert cosines.min() >= 0.99, f"min cosine {cosines.min():.5f} < 0.99"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_moondream_caption_and_ocr_contracts() -> None:
    model = Moondream2(
        VisionModelConfig(
            model_id=settings.vision_model,
            revision=settings.vision_model_revision,
            device="cuda",
            compile_model=False,
        )
    )

    encoded_image = model.model.encode_image(_sample_image())
    caption = model.model.caption(encoded_image, length="short")["caption"]
    ocr_text = model.model.query(encoded_image, model._ocr_prompt, reasoning=False)["answer"]

    assert isinstance(caption, str)
    assert caption.strip()
    assert isinstance(ocr_text, str)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_local_gpu_model_switch_releases_previous_model() -> None:
    moondream = Moondream2(
        VisionModelConfig(
            model_id=settings.vision_model,
            revision=settings.vision_model_revision,
            device="cuda",
            compile_model=False,
        )
    )
    encoded_image = moondream.model.encode_image(_sample_image())
    caption = moondream.model.caption(encoded_image, length="short")["caption"]
    assert caption.strip()

    del encoded_image
    del moondream
    Moondream2.release_instance()

    embedder = SiglipEmbedder(
        EmbedderConfig(
            image_model=settings.embed_model,
            device=settings.embed_device,
            use_bnb_4bit=False,
            fp16_fallback=True,
        )
    )
    image_features = embedder.encode_images([_sample_image()])
    text_features = embedder.encode_texts(["MOONDREAM TEST yellow circle blue rectangle"])

    _assert_embedding_contract(image_features, (1, 768))
    _assert_embedding_contract(text_features, (1, 768))
