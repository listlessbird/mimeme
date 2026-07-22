from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from PIL import Image

pytestmark = pytest.mark.model_smoke


def _enabled() -> bool:
    return os.environ.get("RUN_MODEL_SMOKE") == "1"


@pytest.fixture()
def _guard() -> None:
    if not _enabled():
        pytest.skip("set RUN_MODEL_SMOKE=1 (with local-gpu extras) to run model smoke tests")


def _png(path: Path) -> Path:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 120, 200)).save(buffer, format="PNG")
    path.write_bytes(buffer.getvalue())
    return path


def test_inference_child_embed_smoke(_guard: None, tmp_path: Path) -> None:
    from mimeme.compute.inference import Models
    from mimeme.compute.model import EmbedCall, EmbedCallItem
    from mimeme.shared.config import InferenceConfig

    models = Models(InferenceConfig(embed_device="cpu"))
    path = _png(tmp_path / "img.png")
    reply = models.embed(
        EmbedCall(
            items=[
                EmbedCallItem(
                    image_id=1,
                    path=str(path),
                    text="a blue square",
                    image_out=str(tmp_path / "img.npy"),
                    text_out=str(tmp_path / "txt.npy"),
                )
            ]
        )
    )
    assert reply.items[0].ok
    assert reply.items[0].dimension and reply.items[0].dimension > 0
    assert (tmp_path / "img.npy").exists()
