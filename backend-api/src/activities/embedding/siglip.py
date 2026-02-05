from __future__ import annotations

import threading

import numpy as np
import torch
from PIL.Image import Image
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

from activities.embedding.models import EmbedderConfig


class SiglipEmbedder:
    _instance: SiglipEmbedder | None = None
    _lock = threading.Lock()

    def __init__(self, config: EmbedderConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.image_model_name = config.image_model

        self._load_models()

    @classmethod
    def get_instance(cls, config: EmbedderConfig | None = None) -> SiglipEmbedder:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cfg = config or EmbedderConfig()
                    cls._instance = cls(cfg)
        assert cls._instance is not None
        return cls._instance

    def _load_models(self) -> None:
        quant_cfg = None

        if self.config.use_bnb_4bit:
            quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

        try:
            if quant_cfg:
                self.processor = AutoProcessor.from_pretrained(
                    self.image_model_name, trust_remote_code=True
                )

                self.model = AutoModel.from_pretrained(
                    self.image_model_name,
                    trust_remote_code=True,
                    quantization_config=quant_cfg,
                    device_map="auto",
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                )

            else:
                dtype = torch.float16 if self.config.fp16_fallback else torch.float32

                self.processor = AutoProcessor.from_pretrained(
                    self.image_model_name,
                    trust_remote_code=True,
                    dtype=(torch.float16 if dtype == torch.float16 else None),
                    device_map="auto" if self.device.type == "cuda" else None,
                )

        except Exception:
            # try with siglip
            self.image_model_name = "google/siglip-so400m-patch14-384"
            self.processor = AutoProcessor.from_pretrained(self.image_model_name)
            self.model = AutoModel.from_pretrained(self.image_model_name)

        self.has_get_image_features = hasattr(self.model, "get_image_features")
        self.has_get_text_features = hasattr(self.model, "get_text_features")
        self.is_siglip2 = "siglip2" in self.image_model_name.lower()
        self.is_naflex = "naflex" in self.image_model_name.lower()

    def encode_images(self, images: list[Image]) -> np.ndarray:
        if self.is_siglip2:
            if self.is_naflex:
                inputs = self.processor(
                    images=images,
                    return_tensors="pt",
                    padding="max_length",
                    max_num_patches=256,
                )
            else:
                inputs = self.processor(
                    images=images,
                    return_tensors="pt",
                    padding="max_length",
                )
        else:
            inputs = self.processor(images=images, return_tensors="pt", padding="max_length")

        model_dtype = self.model.dtype

        new_inputs = {}

        for k, v in inputs.items():
            if v.dtype == torch.float32:
                new_inputs[k] = v.to(
                    device=self.model.device,
                    dtype=model_dtype,
                )
            else:
                new_inputs[k] = v.to(self.model.device)

        inputs = new_inputs

        with torch.no_grad():
            if self.has_get_image_features:
                feats = self.model.get_image_features(**inputs)
            else:
                out = self.model(**inputs)
                if hasattr(out, "image_embeds"):
                    feats = out.image_embeds
                elif hasattr(out, "last_hidden_state"):
                    feats = out.last_hidden_state[:, 0, :]
                else:
                    raise ValueError("Couldnt find image features in the model output")
        return feats.cpu().numpy()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if self.is_siglip2:
            texts = [text.lower() for text in texts]

        if self.is_siglip2:
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding="max_length",
                max_length=64,
            )

        else:
            inputs = self.processor(text=texts, return_tensors="pt", padding="max_length")

        model_dtype = self.model.dtype

        new_inputs = {}

        for k, v in inputs.items():
            if v.dtype == torch.float32:
                new_inputs[k] = v.to(
                    device=self.model.device,
                    dtype=model_dtype,
                )
            else:
                new_inputs[k] = v.to(self.model.device)

        inputs = new_inputs

        with torch.no_grad():
            if self.has_get_text_features:
                feats = self.model.get_text_features(**inputs)
            else:
                out = self.model(**inputs)
                if hasattr(out, "text_embeds"):
                    feats = out.text_embeds
                elif hasattr(out, "pooler_output"):
                    feats = out.pooler_output
                elif hasattr(out, "last_hidden_state"):
                    feats = out.last_hidden_state[:, 0, :]
                else:
                    raise ValueError("Unknown model output format")

        return feats.cpu().numpy()

    @property
    def dimension(self) -> int:
        d = self.encode_texts(["test"])
        return int(d.shape[-1])
