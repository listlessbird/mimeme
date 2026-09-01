from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from mimeme.compute.inference import Models
from mimeme.compute.model import AnnotateCall, EmbedCall, EmbedCallItem
from mimeme.config import InferenceConfig


class _VisionLoader:
    loaded: object | None = None

    @staticmethod
    def from_pretrained(*args, **kwargs) -> object:  # noqa: ANN002, ANN003
        _VisionLoader.loaded = _CompilableVisionModel()
        return _VisionLoader.loaded


class _CompilableVisionModel:
    def __init__(self) -> None:
        self.compiled = False

    def compile(self) -> None:
        self.compiled = True


class _VisionModel:
    def encode_image(self, image: Image.Image) -> object:
        return object()

    def caption(self, encoded: object, *, length: str) -> dict[str, str]:
        return {"caption": "a square"}

    def query(self, encoded: object, prompt: str, *, reasoning: bool) -> dict[str, str]:
        return {"answer": "text"}


def test_both_residency_keeps_embed_model_when_loading_vision(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=_VisionLoader),
    )
    models = Models(InferenceConfig(embed_device="cpu", residency="both"))
    embed_model = object()
    models._siglip_model = embed_model

    models._load_vision()

    assert models._siglip_model is embed_model


def test_swap_residency_releases_embed_model_when_loading_vision(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=_VisionLoader),
    )
    models = Models(InferenceConfig(embed_device="cpu", residency="swap"))
    models._siglip_model = object()

    models._load_vision()

    assert models._siglip_model is None


def test_vision_compile_uses_moondream_native_compile(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=_VisionLoader),
    )
    models = Models(InferenceConfig(embed_device="cpu", vision_compile=True))

    loaded = models._load_vision()

    assert isinstance(loaded, _CompilableVisionModel)
    assert loaded.compiled


def test_annotate_reports_phase_timings(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(path)
    models = Models(InferenceConfig(embed_device="cpu", residency="both"))
    models._moondream = _VisionModel()

    reply = models.annotate(AnnotateCall(path=str(path)))

    assert reply.telemetry is not None
    assert reply.telemetry.gpu_model_load_ms == 0
    assert reply.telemetry.image_decode_ms >= 0
    assert reply.telemetry.vision_encode_ms is not None
    assert reply.telemetry.caption_ms is not None
    assert reply.telemetry.ocr_ms is not None
    assert reply.telemetry.residency_mode == "both"


def test_embed_encodes_all_items_in_two_model_batches(tmp_path: Path) -> None:
    paths: list[Path] = []
    for index in range(3):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (8, 8), (index, 0, 0)).save(path)
        paths.append(path)

    models = Models(InferenceConfig(embed_device="cpu"))
    models._load_embed = lambda: None  # type: ignore[method-assign]
    calls: list[tuple[int, int]] = []

    def encode(*, images, texts, telemetry):  # noqa: ANN001, ANN202
        calls.append((len(images or []), len(texts or [])))
        size = len(images) if images is not None else len(texts)
        return np.arange(size * 4, dtype=np.float32).reshape(size, 4)

    models._encode = encode  # type: ignore[method-assign]
    reply = models.embed(
        EmbedCall(
            items=[
                EmbedCallItem(
                    image_id=index,
                    path=str(path),
                    image_out=str(tmp_path / f"image-{index}.npy"),
                )
                for index, path in enumerate(paths)
            ]
        )
    )

    assert calls == [(3, 0)]
    assert all(item.ok for item in reply.items)
    assert reply.telemetry is not None
    assert reply.telemetry.embed_batch_size == 3
    assert reply.telemetry.residency_mode == "both"
    assert np.load(tmp_path / "image-2.npy").tolist() == [8.0, 9.0, 10.0, 11.0]


def test_embed_excludes_invalid_images_from_the_model_batch(tmp_path: Path) -> None:
    valid = tmp_path / "valid.png"
    Image.new("RGB", (8, 8)).save(valid)
    invalid = tmp_path / "invalid.png"
    invalid.write_text("not an image")

    models = Models(InferenceConfig(embed_device="cpu"))
    models._load_embed = lambda: None  # type: ignore[method-assign]
    calls: list[int] = []

    def encode(*, images, texts, telemetry):  # noqa: ANN001, ANN202
        size = len(images) if images is not None else len(texts)
        calls.append(size)
        return np.ones((size, 4), dtype=np.float32)

    models._encode = encode  # type: ignore[method-assign]
    reply = models.embed(
        EmbedCall(
            items=[
                EmbedCallItem(
                    image_id=1,
                    path=str(valid),
                    image_out=str(tmp_path / "valid-image.npy"),
                ),
                EmbedCallItem(
                    image_id=2,
                    path=str(invalid),
                    image_out=str(tmp_path / "invalid-image.npy"),
                ),
            ]
        )
    )

    assert calls == [1]
    assert [item.ok for item in reply.items] == [True, False]


def test_embed_chunks_model_batches_at_the_configured_limit(tmp_path: Path) -> None:
    paths: list[Path] = []
    for index in range(5):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (8, 8)).save(path)
        paths.append(path)

    models = Models(InferenceConfig(embed_device="cpu", embed_batch_size=2))
    models._load_embed = lambda: None  # type: ignore[method-assign]
    calls: list[tuple[int, int]] = []

    def encode(*, images, texts, telemetry):  # noqa: ANN001, ANN202
        calls.append((len(images or []), len(texts or [])))
        size = len(images) if images is not None else len(texts)
        return np.ones((size, 4), dtype=np.float32)

    models._encode = encode  # type: ignore[method-assign]
    reply = models.embed(
        EmbedCall(
            items=[
                EmbedCallItem(
                    image_id=index,
                    path=str(path),
                    image_out=str(tmp_path / f"image-{index}.npy"),
                )
                for index, path in enumerate(paths)
            ]
        )
    )

    assert calls == [(2, 0), (2, 0), (1, 0)]
    assert all(item.ok for item in reply.items)
