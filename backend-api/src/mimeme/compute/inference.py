from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Literal, cast

import numpy as np
from PIL import Image

from mimeme.compute.image import prepare_rgb
from mimeme.compute.model import (
    AnnotateCall,
    AnnotateReply,
    EmbedCall,
    EmbedCallItem,
    EmbedReply,
    EmbedReplyItem,
    InferenceTelemetry,
)
from mimeme.config import InferenceConfig
from mimeme.inference.model import caption_prompt


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

    def _sync_cuda(self, torch) -> None:  # noqa: ANN001
        if self._config.embed_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _reset_gpu_peaks(self, torch) -> None:  # noqa: ANN001
        if self._config.embed_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _gpu_peaks(self, torch) -> tuple[float | None, float | None]:  # noqa: ANN001
        if not self._config.embed_device.startswith("cuda") or not torch.cuda.is_available():
            return None, None
        divisor = 1024 * 1024
        return (
            round(torch.cuda.max_memory_allocated() / divisor, 2),
            round(torch.cuda.max_memory_reserved() / divisor, 2),
        )

    def _gpu_device_name(self, torch) -> str | None:  # noqa: ANN001
        if not self._config.embed_device.startswith("cuda") or not torch.cuda.is_available():
            return None
        return str(torch.cuda.get_device_name())

    def _load_vision(self) -> object:
        if self._moondream is None:
            if self._config.residency == "swap":
                self._release_embed()
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                self._config.vision_model,
                revision=self._config.vision_model_revision,
                trust_remote_code=True,
                device_map={"": self._config.embed_device},
            )
            if self._config.vision_compile:
                model.compile()
            self._moondream = model
        return self._moondream

    def _load_embed(self) -> None:
        if self._siglip_model is not None:
            return
        if self._config.residency == "swap":
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
        import torch

        self._reset_gpu_peaks(torch)
        cold_load = self._moondream is None
        load_started = time.perf_counter()
        model = cast("object", self._load_vision())
        self._sync_cuda(torch)
        model_load_ms = round((time.perf_counter() - load_started) * 1000, 2) if cold_load else 0

        decode_started = time.perf_counter()
        with Image.open(call.path) as source:
            image = prepare_rgb(source)
        image_decode_ms = round((time.perf_counter() - decode_started) * 1000, 2)

        self._sync_cuda(torch)
        encode_started = time.perf_counter()
        encoded = model.encode_image(image)  # type: ignore[attr-defined]
        self._sync_cuda(torch)
        vision_encode_ms = round((time.perf_counter() - encode_started) * 1000, 2)

        caption_started = time.perf_counter()
        if call.context is None:
            caption = model.caption(encoded, length=call.length)["caption"]  # type: ignore[attr-defined]
        else:
            caption = model.query(  # type: ignore[attr-defined]
                encoded,
                caption_prompt(call.context),
                reasoning=False,
            )["answer"]
        self._sync_cuda(torch)
        caption_ms = round((time.perf_counter() - caption_started) * 1000, 2)

        ocr_started = time.perf_counter()
        ocr = model.query(  # type: ignore[attr-defined]
            encoded,
            "Transcribe the text in natural reading order.",
            reasoning=False,
        )["answer"]
        self._sync_cuda(torch)
        ocr_ms = round((time.perf_counter() - ocr_started) * 1000, 2)
        peak_allocated, peak_reserved = self._gpu_peaks(torch)
        version = self._vision_version
        return AnnotateReply(
            caption=caption,
            caption_model=version,
            ocr_text=ocr,
            ocr_model=version,
            telemetry=InferenceTelemetry(
                gpu_model_load_ms=model_load_ms,
                image_decode_ms=image_decode_ms,
                vision_encode_ms=vision_encode_ms,
                caption_ms=caption_ms,
                ocr_ms=ocr_ms,
                gpu_peak_allocated_mb=peak_allocated,
                gpu_peak_reserved_mb=peak_reserved,
                gpu_device_name=self._gpu_device_name(torch),
                residency_mode=self._config.residency,
            ),
        )

    def _encode(
        self,
        *,
        images: list[Image.Image] | None,
        texts: list[str] | None,
        telemetry: dict[str, float],
    ) -> np.ndarray:
        import torch

        assert self._siglip_model is not None and self._siglip_processor is not None
        assert self._siglip_flags is not None
        has_image, has_text, is_siglip2, is_naflex = self._siglip_flags
        processor = self._siglip_processor
        model = self._siglip_model

        preprocess_started = time.perf_counter()
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
        self._sync_cuda(torch)
        telemetry["siglip_preprocess_ms"] += round(
            (time.perf_counter() - preprocess_started) * 1000, 2
        )

        self._sync_cuda(torch)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            if kind == "image" and has_image:
                out = model.get_image_features(**moved)  # type: ignore[attr-defined]
            elif kind == "text" and has_text:
                out = model.get_text_features(**moved)  # type: ignore[attr-defined]
            else:
                out = model(**moved)  # type: ignore[operator]
        self._sync_cuda(torch)
        telemetry[f"siglip_{kind}_ms"] += round((time.perf_counter() - inference_started) * 1000, 2)
        return _to_numpy(out, kind=kind)

    def embed(self, call: EmbedCall) -> EmbedReply:
        import torch

        self._reset_gpu_peaks(torch)
        cold_load = self._siglip_model is None
        load_started = time.perf_counter()
        self._load_embed()
        self._sync_cuda(torch)
        model_load_ms = round((time.perf_counter() - load_started) * 1000, 2) if cold_load else 0
        model_name = self._config.embed_model
        items: list[EmbedReplyItem | None] = [None] * len(call.items)
        prepared: list[tuple[int, EmbedCallItem, Image.Image]] = []
        image_decode_ms = 0.0
        for index, item in enumerate(call.items):
            try:
                decode_started = time.perf_counter()
                with Image.open(item.path) as source:
                    image = prepare_rgb(source)
                image_decode_ms += round((time.perf_counter() - decode_started) * 1000, 2)
                prepared.append((index, item, image))
            except Exception as exc:
                items[index] = EmbedReplyItem(image_id=item.image_id, ok=False, error=str(exc))

        batch_size = self._config.embed_batch_size
        forward_batch_size = 0
        timing = {
            "siglip_preprocess_ms": 0.0,
            "siglip_image_ms": 0.0,
        }
        for offset in range(0, len(prepared), batch_size):
            chunk = prepared[offset : offset + batch_size]
            forward_batch_size = max(forward_batch_size, len(chunk))
            try:
                image_feats = self._encode(
                    images=[image for _, _, image in chunk], texts=None, telemetry=timing
                )
            except Exception as exc:
                for index, item, _ in chunk:
                    items[index] = EmbedReplyItem(image_id=item.image_id, ok=False, error=str(exc))
            else:
                for row, (index, item, _) in enumerate(chunk):
                    try:
                        _save_npy(Path(item.image_out), image_feats[row])
                        items[index] = EmbedReplyItem(
                            image_id=item.image_id,
                            ok=True,
                            model=model_name,
                            dimension=int(image_feats.shape[-1]),
                        )
                    except Exception as exc:
                        items[index] = EmbedReplyItem(
                            image_id=item.image_id, ok=False, error=str(exc)
                        )

        assert all(item is not None for item in items)
        peak_allocated, peak_reserved = self._gpu_peaks(torch)
        return EmbedReply(
            items=[item for item in items if item is not None],
            telemetry=InferenceTelemetry(
                gpu_model_load_ms=model_load_ms,
                image_decode_ms=round(image_decode_ms, 2),
                siglip_preprocess_ms=round(timing["siglip_preprocess_ms"], 2),
                siglip_image_ms=round(timing["siglip_image_ms"], 2),
                embed_batch_size=forward_batch_size,
                gpu_peak_allocated_mb=peak_allocated,
                gpu_peak_reserved_mb=peak_reserved,
                gpu_device_name=self._gpu_device_name(torch),
                residency_mode=self._config.residency,
            ),
        )


def _save_npy(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, array)
        handle.flush()
