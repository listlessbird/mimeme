from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from mimeme.compute.inference import Models
from mimeme.compute.model import EmbedCall, EmbedCallItem
from mimeme.config import InferenceConfig


def test_embed_encodes_all_items_in_two_model_batches(tmp_path: Path) -> None:
    paths: list[Path] = []
    for index in range(3):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (8, 8), (index, 0, 0)).save(path)
        paths.append(path)

    models = Models(InferenceConfig(embed_device="cpu"))
    models._load_embed = lambda: None  # type: ignore[method-assign]
    calls: list[tuple[int, int]] = []

    def encode(*, images, texts):  # noqa: ANN001, ANN202
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
                    text=f"text {index}",
                    image_out=str(tmp_path / f"image-{index}.npy"),
                    text_out=str(tmp_path / f"text-{index}.npy"),
                )
                for index, path in enumerate(paths)
            ]
        )
    )

    assert calls == [(3, 0), (0, 3)]
    assert all(item.ok for item in reply.items)
    assert np.load(tmp_path / "image-2.npy").tolist() == [8.0, 9.0, 10.0, 11.0]


def test_embed_excludes_invalid_images_from_the_model_batch(tmp_path: Path) -> None:
    valid = tmp_path / "valid.png"
    Image.new("RGB", (8, 8)).save(valid)
    invalid = tmp_path / "invalid.png"
    invalid.write_text("not an image")

    models = Models(InferenceConfig(embed_device="cpu"))
    models._load_embed = lambda: None  # type: ignore[method-assign]
    calls: list[int] = []

    def encode(*, images, texts):  # noqa: ANN001, ANN202
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
                    text="valid",
                    image_out=str(tmp_path / "valid-image.npy"),
                    text_out=str(tmp_path / "valid-text.npy"),
                ),
                EmbedCallItem(
                    image_id=2,
                    path=str(invalid),
                    text="invalid",
                    image_out=str(tmp_path / "invalid-image.npy"),
                    text_out=str(tmp_path / "invalid-text.npy"),
                ),
            ]
        )
    )

    assert calls == [1, 1]
    assert [item.ok for item in reply.items] == [True, False]
