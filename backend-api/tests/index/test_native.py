from __future__ import annotations

import json

import faiss  # type: ignore[import-untyped]
import numpy as np

from mimeme import index
from mimeme.index.native import build


def test_native_builder_writes_retrievable_normalized_generation(tmp_path) -> None:
    inputs = tmp_path / "inputs"
    output = tmp_path / "output"
    inputs.mkdir()
    np.save(inputs / "1.npy", np.array([3.0, 0.0], dtype=np.float32))
    np.save(inputs / "2.npy", np.array([0.0, 4.0], dtype=np.float32))

    result = build(
        index.PreparedBuild(
            version="v2-test",
            target_generation=4,
            model="test/embed",
            index_type="flat",
            dimension=2,
            encoder=index.Encoder(repo="test/encoder", revision="rev", variant="model.onnx"),
            output_dir=str(output),
            embeddings=[
                index.LocalEmbedding(image_id=1, image_path=str(inputs / "1.npy")),
                index.LocalEmbedding(image_id=2, image_path=str(inputs / "2.npy")),
            ],
        )
    )

    built = faiss.read_index(str(output / "index.faiss"))
    scores, rows = built.search(np.array([[1.0, 0.0]], dtype=np.float32), 2)
    mapping = json.loads((output / "mapping.json").read_text())
    assert mapping[str(rows[0][0])] == 1
    assert scores[0][0] == 1.0
    assert {file.name for file in result.files} == {
        "index.faiss",
        "mapping.json",
        "metadata.json",
    }
