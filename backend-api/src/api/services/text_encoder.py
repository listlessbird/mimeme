from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import structlog
import torch
from transformers import AutoModel, AutoProcessor

from shared.config import settings

_log = structlog.get_logger().bind(component="search_text_encoder")


class SearchTextEncoder:
    """Lightweight local text encoder for search queries.

    Loads the SigLIP model once on CPU so search queries bypass
    the Temporal → Modal round-trip entirely.
    """

    _instance: SearchTextEncoder | None = None
    _lock = threading.Lock()

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self._is_siglip2 = "siglip2" in model_name.lower()

        started = time.monotonic()
        _log.info(
            "text_encoder_loading",
            model=model_name,
            device=device,
        )

        self._processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to(self.device)
        self._model.eval()

        duration_ms = int((time.monotonic() - started) * 1000)
        _log.info(
            "text_encoder_ready",
            model=model_name,
            device=device,
            duration_ms=duration_ms,
        )

    @classmethod
    def get_instance(cls) -> SearchTextEncoder:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        model_name=settings.embed_model,
                        device=settings.search_text_encoder_device,
                    )
        assert cls._instance is not None
        return cls._instance

    def encode(self, query: str) -> np.ndarray:
        """Encode a single text query into an embedding vector.

        Returns the raw (unnormalised) embedding — L2 normalisation is
        handled by FaissIndexManager.search() before querying the index.
        """
        text = query.lower() if self._is_siglip2 else query

        if self._is_siglip2:
            inputs = self._processor(
                text=[text],
                return_tensors="pt",
                padding="max_length",
                max_length=64,
            )
        else:
            inputs = self._processor(
                text=[text],
                return_tensors="pt",
                padding="max_length",
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if hasattr(self._model, "get_text_features"):
                feats = self._extract_tensor_features(self._model.get_text_features(**inputs))
            else:
                out = self._model(**inputs)
                feats = self._extract_tensor_features(out)

        return feats.detach().cpu().numpy().astype(np.float32)[0]

    def _extract_tensor_features(self, out: Any) -> torch.Tensor:
        if isinstance(out, torch.Tensor):
            return out

        if hasattr(out, "text_embeds"):
            return out.text_embeds
        if hasattr(out, "pooler_output"):
            return out.pooler_output
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0, :]

        raise ValueError("Unknown model output format")
