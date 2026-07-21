from __future__ import annotations

import gc
import threading
import time

import numpy as np
import structlog
import torch
from PIL.Image import Image
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

from mimeme.activities.embedding.models import EmbedderConfig
from mimeme.domain.inference import select_pooled_feature_tensor


class SiglipEmbedder:
    _instance: SiglipEmbedder | None = None
    _lock = threading.Lock()
    _log = structlog.get_logger().bind(component="siglip_embedder")

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

    @classmethod
    def release_instance(cls) -> None:
        with cls._lock:
            if cls._instance is None:
                return
            cls._log.info("embedding_step", step="singleton_instance_released")
            cls._instance = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _load_models(self) -> None:
        started = time.monotonic()
        outcome = "success"
        error_type: str | None = None
        error_message: str | None = None
        log = self._log.bind(
            model_name=self.image_model_name,
            device=str(self.device),
            use_bnb_4bit=self.config.use_bnb_4bit,
        )
        quant_cfg = None

        if self.config.use_bnb_4bit:
            quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

        try:
            log.info("embedding_step", step="model_load_primary_start")
            if quant_cfg:
                self.processor = AutoProcessor.from_pretrained(
                    self.image_model_name, trust_remote_code=True
                )

                self.model = AutoModel.from_pretrained(
                    self.image_model_name,
                    trust_remote_code=True,
                    quantization_config=quant_cfg,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
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
                self.model = AutoModel.from_pretrained(
                    self.image_model_name,
                    trust_remote_code=True,
                    device_map="auto" if self.device.type == "cuda" else None,
                    torch_dtype=(torch.float16 if dtype == torch.float16 else None),
                    attn_implementation="sdpa" if self.device.type == "cuda" else None,
                )
            log.info("embedding_step", step="model_load_primary_complete")

            self.has_get_image_features = hasattr(self.model, "get_image_features")
            self.has_get_text_features = hasattr(self.model, "get_text_features")
            self.is_siglip2 = "siglip2" in self.image_model_name.lower()
            self.is_naflex = "naflex" in self.image_model_name.lower()
        except Exception as exc:
            outcome = "error"
            error_type = type(exc).__name__
            error_message = str(exc)
            log.error(
                "embedding_step",
                step="model_init_failed",
                error_type=error_type,
                error=error_message,
                exc_info=True,
            )
            raise
        finally:
            log.info(
                "embedding_wide_event",
                event_type="embedding_wide_event",
                phase="model_init",
                outcome=outcome,
                duration_ms=int((time.monotonic() - started) * 1000),
                model_name=self.image_model_name,
                error_type=error_type,
                error=error_message,
            )

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
                feats = select_pooled_feature_tensor(
                    self.model.get_image_features(**inputs),
                    kind="image",
                )
            else:
                out = self.model(**inputs)
                feats = select_pooled_feature_tensor(out, kind="image")
        return feats.cpu().numpy()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
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
                feats = select_pooled_feature_tensor(
                    self.model.get_text_features(**inputs),
                    kind="text",
                )
            else:
                out = self.model(**inputs)
                feats = select_pooled_feature_tensor(out, kind="text")

        return feats.cpu().numpy()

    @property
    def dimension(self) -> int:
        d = self.encode_texts(["test"])
        return int(d.shape[-1])
