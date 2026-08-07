from __future__ import annotations

import gc
from pathlib import Path
from typing import Literal, cast

import numpy as np
from PIL import Image

from mimeme.compute.image import prepare_rgb
from mimeme.compute.model import (
    AnnotateCall,
    AnnotateReply,
    EmbedCall,
    EmbedReply,
    EmbedReplyItem,
)
from mimeme.config import InferenceConfig


def _to_numpy(output: object, *, kind: Literal["image", "text"]) -> np.ndarray:
    if hasattr(output, "cpu"):
        tensor = output
    else:
        fields = (
            ("image_embeds", "pooler_output")
            if kind == "image"
            else (
                "text_embeds",
                "pooler_output",
            )
        )
        tensor = None
        for field in fields:
            value = getattr(output, field, None)
            if value is not None and hasattr(value, "cpu"):
                tensor = value
                break
        if tensor is None:
            raise ValueError(f"unknown {kind} model output format")
    array = cast(np.ndarray, tensor.cpu().numpy())  # type: ignore[union-attr]
    return array.astype(np.float32, copy=False)


class Models:
    def __init__(self, config: InferenceConfig) -> None:
        self._config = config
        self._moondream: object | None = None
        self._siglip_model: object | None = None
        self._siglip_processor: object | None = None
        self._siglip_flags: tuple[bool, bool, bool, bool] | None = None

    @property
    def _vision_version(self) -> str:
        rev = self._config.vision_model_revision or "latest"
        return f"{self._config.vision_model}@{rev}"

    def _release_vision(self) -> None:
        self._moondream = None
        gc.collect()

    def _release_embed(self) -> None:
        self._siglip_model = None
        self._siglip_processor = None
        self._siglip_flags = None
        gc.collect()

    def _load_vision(self) -> object:
        if self._moondream is None:
            self._release_embed()
            from transformers import AutoModelForCausalLM

            self._moondream = AutoModelForCausalLM.from_pretrained(
                self._config.vision_model,
                revision=self._config.vision_model_revision,
                trust_remote_code=True,
                device_map={"": self._config.embed_device},
            )
        return self._moondream

    def _load_embed(self) -> None:
        if self._siglip_model is not None:
            return
        self._release_vision()
        import torch
        from transformers import AutoModel, AutoProcessor

        name = self._config.embed_model
        device = self._config.embed_device
        use_cuda = device == "cuda"
        self._siglip_processor = AutoProcessor.from_pretrained(
            name,
            trust_remote_code=True,
            dtype=torch.float16 if use_cuda else None,
            device_map="auto" if use_cuda else None,
        )
        self._siglip_model = AutoModel.from_pretrained(
            name,
            trust_remote_code=True,
            device_map="auto" if use_cuda else None,
            torch_dtype=torch.float16 if use_cuda else None,
            attn_implementation="sdpa" if use_cuda else None,
        )
        model = self._siglip_model
        self._siglip_flags = (
            hasattr(model, "get_image_features"),
            hasattr(model, "get_text_features"),
            "siglip2" in name.lower(),
            "naflex" in name.lower(),
        )

    def annotate(self, call: AnnotateCall) -> AnnotateReply:
        model = cast("object", self._load_vision())
        image = prepare_rgb(Image.open(call.path))
        encoded = model.encode_image(image)  # type: ignore[attr-defined]
        caption = model.caption(encoded, length=call.length)["caption"]  # type: ignore[attr-defined]
        ocr = model.query(  # type: ignore[attr-defined]
            encoded,
            "Transcribe the text in natural reading order.",
            reasoning=False,
        )["answer"]
        version = self._vision_version
        return AnnotateReply(
            caption=caption,
            caption_model=version,
            ocr_text=ocr,
            ocr_model=version,
        )

    def _encode(self, *, images: list[Image.Image] | None, texts: list[str] | None) -> np.ndarray:
        import torch

        assert self._siglip_model is not None and self._siglip_processor is not None
        assert self._siglip_flags is not None
        has_image, has_text, is_siglip2, is_naflex = self._siglip_flags
        processor = self._siglip_processor
        model = self._siglip_model

        if images is not None:
            if is_siglip2 and is_naflex:
                inputs = processor(  # type: ignore[operator]
                    images=images,
                    return_tensors="pt",
                    padding="max_length",
                    max_num_patches=256,
                )
            else:
                inputs = processor(  # type: ignore[operator]
                    images=images, return_tensors="pt", padding="max_length"
                )
            kind: Literal["image", "text"] = "image"
        else:
            if is_siglip2:
                inputs = processor(  # type: ignore[operator]
                    text=texts, return_tensors="pt", padding="max_length", max_length=64
                )
            else:
                inputs = processor(  # type: ignore[operator]
                    text=texts, return_tensors="pt", padding="max_length"
                )
            kind = "text"

        moved = {}
        for key, value in inputs.items():
            if value.dtype == torch.float32:
                moved[key] = value.to(device=model.device, dtype=model.dtype)  # type: ignore[attr-defined]
            else:
                moved[key] = value.to(model.device)  # type: ignore[attr-defined]

        with torch.no_grad():
            if kind == "image" and has_image:
                out = model.get_image_features(**moved)  # type: ignore[attr-defined]
            elif kind == "text" and has_text:
                out = model.get_text_features(**moved)  # type: ignore[attr-defined]
            else:
                out = model(**moved)  # type: ignore[operator]
        return _to_numpy(out, kind=kind)

    def embed(self, call: EmbedCall) -> EmbedReply:
        self._load_embed()
        model_name = self._config.embed_model
        items: list[EmbedReplyItem] = []
        for item in call.items:
            try:
                image = prepare_rgb(Image.open(item.path))
                image_feats = self._encode(images=[image], texts=None)
                text_feats = self._encode(images=None, texts=[item.text])
                _save_npy(Path(item.image_out), image_feats[0])
                _save_npy(Path(item.text_out), text_feats[0])
                items.append(
                    EmbedReplyItem(
                        image_id=item.image_id,
                        ok=True,
                        model=model_name,
                        dimension=int(image_feats.shape[-1]),
                    )
                )
            except Exception as exc:
                items.append(EmbedReplyItem(image_id=item.image_id, ok=False, error=str(exc)))
        return EmbedReply(items=items)


def _save_npy(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, array)
        handle.flush()
