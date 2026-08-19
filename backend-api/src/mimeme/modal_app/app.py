from __future__ import annotations

import io
from typing import Any, Literal, cast

import modal

from mimeme import release
from mimeme.config import ArtifactConfig, MediaConfig, Settings
from mimeme.inference.model import ANNOTATION_CONTRACT_VERSION

settings = Settings()
app = modal.App(settings.compute.modal_app_name)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "libvips42", "libjpeg-dev", "libtiff-dev")
    .pip_install(
        "torch==2.9.0",
        "torchvision==0.24.0",
        "transformers==4.52.4",
        "accelerate==1.12.0",
        "einops==0.8.2",
        "pillow==12.1.1",
        "numpy==2.4.2",
        "aiobotocore>=3.7.0",
        "bitsandbytes==0.48.2",
        "pyvips==3.1.1",
        "structlog==25.5.0",
        "pydantic==2.12.5",
        "pydantic-settings==2.13.1",
    )
    .add_local_python_source("mimeme")
)

hf_cache = modal.Volume.from_name(
    settings.compute.modal_hf_cache_volume_name, create_if_missing=True
)
HF_CACHE_DIR = "/root/.cache/huggingface"
s3_secret = modal.Secret.from_name(settings.compute.modal_s3_secret_name)


@app.function()
def release_info() -> dict[str, str | int]:
    return {
        "release_id": release.ID,
        "annotation_contract_version": ANNOTATION_CONTRACT_VERSION,
    }


def _prepare_rgb(image: Any) -> Any:
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    return image.convert("RGB")


def _to_numpy(output: Any, *, kind: Literal["image", "text"]) -> Any:
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
    return tensor.cpu().numpy().astype("float32", copy=False)


async def _open_media():  # noqa: ANN202
    from mimeme import storage

    return await storage.S3.open(_config(Settings().media))


async def _open_artifacts():  # noqa: ANN202
    from mimeme import storage

    return await storage.S3.open(_config(Settings().artifacts))


def _config(config: MediaConfig | ArtifactConfig):  # noqa: ANN202
    from mimeme import storage

    return storage.Config(
        endpoint_url=config.s3_endpoint_url,
        region=config.s3_region,
        access_key=config.s3_access_key_id,
        secret_key=config.s3_secret_access_key,
        bucket=config.s3_bucket,
        force_path_style=config.s3_force_path_style,
    )


@app.cls(
    image=gpu_image,
    gpu="T4",
    volumes={HF_CACHE_DIR: hf_cache},
    secrets=[s3_secret],
    scaledown_window=300,
)
class VisionService:
    model_version: str = "vikhyatk/moondream2@2025-06-21"

    @modal.enter()
    def load_model(self) -> None:
        from transformers import AutoModelForCausalLM

        self._model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-06-21",
            trust_remote_code=True,
            device_map={"": "cuda"},
        )

    @modal.method()
    async def annotate_image(
        self, media_key: str, length: str = "normal", context: dict | None = None
    ) -> dict:
        from PIL import Image

        from mimeme import storage

        media = await _open_media()
        try:
            data = await media.read_bytes(storage.Object(media_key), max_bytes=64 * 1024 * 1024)
        finally:
            await media.close()

        pil = _prepare_rgb(Image.open(io.BytesIO(data)))
        model = cast(Any, self._model)
        encoded = model.encode_image(pil)
        if context is None:
            caption = model.caption(encoded, length=length)["caption"]
        else:
            from mimeme.inference.model import Context, caption_prompt

            caption = model.query(
                encoded,
                caption_prompt(Context.model_validate(context)),
                reasoning=False,
            )["answer"]
        ocr_text = model.query(
            encoded, "Transcribe the text in natural reading order.", reasoning=False
        )["answer"]
        return {
            "caption": caption,
            "caption_model": self.model_version,
            "ocr_text": ocr_text,
            "ocr_model": self.model_version,
        }


@app.cls(
    image=gpu_image,
    gpu="T4",
    volumes={HF_CACHE_DIR: hf_cache},
    secrets=[s3_secret],
    scaledown_window=300,
)
class EmbeddingService:
    model_name: str = "google/siglip2-base-patch16-naflex"

    @modal.enter()
    def load_model(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True, dtype=torch.float16, device_map="auto"
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )
        self._is_siglip2 = "siglip2" in self.model_name.lower()
        self._is_naflex = "naflex" in self.model_name.lower()
        self._has_image = hasattr(self.model, "get_image_features")
        self._has_text = hasattr(self.model, "get_text_features")

    def _to_device(self, inputs: dict) -> dict:
        import torch

        out = {}
        for key, value in inputs.items():
            if value.dtype == torch.float32:
                out[key] = value.to(device=self.model.device, dtype=self.model.dtype)
            else:
                out[key] = value.to(self.model.device)
        return out

    def _encode_images(self, images: list) -> Any:
        import torch

        if self._is_naflex:
            inputs = self.processor(
                images=images, return_tensors="pt", padding="max_length", max_num_patches=256
            )
        else:
            inputs = self.processor(images=images, return_tensors="pt", padding="max_length")
        inputs = self._to_device(inputs)
        with torch.no_grad():
            feats = (
                self.model.get_image_features(**inputs) if self._has_image else self.model(**inputs)
            )
        return _to_numpy(feats, kind="image")

    def _encode_texts(self, texts: list[str]) -> Any:
        import torch

        if self._is_siglip2:
            inputs = self.processor(
                text=texts, return_tensors="pt", padding="max_length", max_length=64
            )
        else:
            inputs = self.processor(text=texts, return_tensors="pt", padding="max_length")
        inputs = self._to_device(inputs)
        with torch.no_grad():
            feats = (
                self.model.get_text_features(**inputs) if self._has_text else self.model(**inputs)
            )
        return _to_numpy(feats, kind="text")

    @modal.method()
    async def embed_batch(self, items: list[dict], model: str) -> dict:
        import numpy as np
        from PIL import Image

        from mimeme import storage

        media = await _open_media()
        artifacts = await _open_artifacts()
        try:
            results: list[dict | None] = [None] * len(items)
            prepared: list[tuple[int, dict, Any]] = []
            for index, item in enumerate(items):
                try:
                    data = await media.read_bytes(
                        storage.Object(item["media_key"]), max_bytes=64 * 1024 * 1024
                    )
                    pil = _prepare_rgb(Image.open(io.BytesIO(data)))
                    prepared.append((index, item, pil))
                except Exception as exc:
                    results[index] = {
                        "image_id": item["image_id"],
                        "ok": False,
                        "error": str(exc),
                    }

            if prepared:
                try:
                    img_feats = self._encode_images([pil for _, _, pil in prepared])
                    txt_feats = self._encode_texts([item["text"] for _, item, _ in prepared])
                except Exception as exc:
                    for index, item, _ in prepared:
                        results[index] = {
                            "image_id": item["image_id"],
                            "ok": False,
                            "error": str(exc),
                        }
                else:
                    for row, (index, item, _) in enumerate(prepared):
                        try:
                            await artifacts.put_bytes(
                                storage.Object(item["image_key"]),
                                _npy_bytes(np, img_feats[row]),
                                content_type="application/octet-stream",
                            )
                            await artifacts.put_bytes(
                                storage.Object(item["text_key"]),
                                _npy_bytes(np, txt_feats[row]),
                                content_type="application/octet-stream",
                            )
                            results[index] = {
                                "image_id": item["image_id"],
                                "ok": True,
                                "image_key": item["image_key"],
                                "text_key": item["text_key"],
                                "model": model,
                                "dimension": int(img_feats.shape[-1]),
                            }
                        except Exception as exc:
                            results[index] = {
                                "image_id": item["image_id"],
                                "ok": False,
                                "error": str(exc),
                            }
            assert all(result is not None for result in results)
            return {"items": [result for result in results if result is not None]}
        finally:
            await artifacts.close()
            await media.close()


def _npy_bytes(np: Any, array: Any) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()
