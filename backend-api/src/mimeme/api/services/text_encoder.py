from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort
import structlog
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from mimeme.shared.runtime import settings

_log = structlog.get_logger().bind(component="search_text_encoder")


class SearchTextEncoder:
    _instance: SearchTextEncoder | None = None
    _lock = threading.Lock()

    def __init__(self, repo_id: str, revision: str, variant: str, threads: int) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.variant = variant

        started = time.monotonic()
        _log.info(
            "text_encoder_loading",
            repo=repo_id,
            revision=revision,
            variant=variant,
            threads=threads,
        )

        artifact_dir = Path(
            snapshot_download(
                repo_id,
                revision=revision,
                allow_patterns=[variant, "tokenizer.json", "export_meta.json"],
            )
        )

        meta = json.loads((artifact_dir / "export_meta.json").read_text())
        self.source_model: str = meta["source_model"]
        max_length = meta.get("max_length", 64)
        pad_token_id = meta.get("pad_token_id", 0)

        self._tokenizer = Tokenizer.from_file(str(artifact_dir / "tokenizer.json"))
        self._tokenizer.enable_padding(length=max_length, pad_id=pad_token_id, pad_token="<pad>")
        self._tokenizer.enable_truncation(max_length=max_length)

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            str(artifact_dir / variant),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        duration_ms = int((time.monotonic() - started) * 1000)
        _log.info(
            "text_encoder_ready",
            repo=repo_id,
            revision=revision,
            variant=variant,
            source_model=self.source_model,
            duration_ms=duration_ms,
        )

    @classmethod
    def get_instance(cls) -> SearchTextEncoder:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        repo_id=settings.inference.onnx_text_encoder_repo,
                        revision=settings.inference.onnx_text_encoder_revision,
                        variant=settings.inference.onnx_text_encoder_variant,
                        threads=settings.inference.onnx_text_encoder_threads,
                    )
        assert cls._instance is not None
        return cls._instance

    def tokenize(self, query: str) -> np.ndarray:
        return np.array([self._tokenizer.encode(query).ids], dtype=np.int64)

    def encode(self, query: str) -> np.ndarray:
        input_ids = self.tokenize(query)
        output = cast(np.ndarray, self._session.run(["text_embeds"], {"input_ids": input_ids})[0])
        return output[0].astype(np.float32)
