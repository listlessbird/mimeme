from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip2-base-patch16-naflex"
MAX_LENGTH = 64
EMBED_DIM = 768


class TorchTextEncoder:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cpu") -> None:
        self.model_id = model_id
        self.device = torch.device(device)
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(self.device)
        self._model.eval()

    def tokenize(self, query: str) -> dict[str, torch.Tensor]:
        return dict(
            self._processor(
                text=[query],
                return_tensors="pt",
                padding="max_length",
                max_length=MAX_LENGTH,
            )
        )

    def encode(self, query: str) -> np.ndarray:
        inputs = {k: v.to(self.device) for k, v in self.tokenize(query).items()}
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        return feats.detach().cpu().numpy().astype(np.float32)[0]
