from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

DEFAULT_MAX_LENGTH = 64
DEFAULT_PAD_TOKEN_ID = 0


class OnnxTextEncoder:
    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path | None = None,
        intra_op_threads: int | None = None,
    ) -> None:
        model_path = Path(model_path)
        tokenizer_path = Path(tokenizer_path or model_path.parent / "tokenizer.json")

        max_length = DEFAULT_MAX_LENGTH
        pad_token_id = DEFAULT_PAD_TOKEN_ID
        meta_path = model_path.parent / "export_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            max_length = meta.get("max_length", max_length)
            pad_token_id = meta.get("pad_token_id", pad_token_id)

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_padding(length=max_length, pad_id=pad_token_id, pad_token="<pad>")
        self._tokenizer.enable_truncation(max_length=max_length)

        options = ort.SessionOptions()
        if intra_op_threads is not None:
            options.intra_op_num_threads = intra_op_threads
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def tokenize(self, query: str) -> np.ndarray:
        return np.array([self._tokenizer.encode(query).ids], dtype=np.int64)

    def encode(self, query: str) -> np.ndarray:
        input_ids = self.tokenize(query)
        output = cast(np.ndarray, self._session.run(["text_embeds"], {"input_ids": input_ids})[0])
        return output[0].astype(np.float32)
