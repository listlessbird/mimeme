from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import boto3
import modal
from botocore.config import Config as BotoConfig

app = modal.App("findmeme-gpu")

gpu_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("libgl1", "libglib2.0-0", "libvips42", "libjpeg-dev", "libtiff-dev")
    .pip_install(
        "torch>=2.3",
        "torchvision>=0.24",
        "transformers>=4.45",
        "accelerate>=0.34",
        "einops>=0.8.1",
        "pillow>=12.0",
        "numpy>=2.0",
        "boto3>=1.40",
        "bitsandbytes>=0.48",
    )
)

hf_cache = modal.Volume.from_name("findmeme-hf-cache", create_if_missing=True)

HF_CACHE_DIR = "/root/.cache/huggingface"

s3_secret = modal.Secret.from_name("findmeme-s3")


def _s3_client():
    session = boto3.Session(
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("S3_REGION", "us-east-1"),
    )
    force_path = os.environ.get("S3_FORCE_PATH_STYLE", "true").lower() == "true"

    return session.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
        config=BotoConfig(
            s3={"addressing_style": "path" if force_path else "auto"},
            signature_version="s3v4",
        ),
    )


def _bucket() -> str:
    return os.environ["S3_BUCKET"]


def _download_image(s3_key: str) -> Path:
    client = _s3_client()
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    client.download_file(_bucket(), s3_key, tmp.name)
    return Path(tmp.name)


def _upload_numpy(arr, s3_key: str) -> None:
    import numpy as np

    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    _s3_client().upload_fileobj(
        buf, _bucket(), s3_key, ExtraArgs={"ContentType": "application/octet-stream"}
    )


@app.cls(
    image=gpu_image,
    gpu="T4",
    volumes={HF_CACHE_DIR: hf_cache},
    secrets=[s3_secret],
    scaledown_window=300,
)
class VisionService:
    model_version: str = "vikhyatk/moondream2@2025-01-09"

    @modal.enter()
    def load_model(self):
        from transformers import AutoModelForCausalLM

        self._model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-01-09",
            trust_remote_code=True,
            device_map={"": "cuda"},
        )
        inner = getattr(self._model, "model", None)
        if inner and hasattr(inner, "compile"):
            inner.compile()

    @modal.method()
    def caption(self, s3_key: str, length: str = "normal") -> dict:
        from PIL import Image

        tmp_path = _download_image(s3_key)
        try:
            pil = Image.open(tmp_path).convert("RGB")
            out = self._model.caption(pil, length)
            cap = out.get("caption", "") if isinstance(out, dict) else out

            if not isinstance(cap, str):
                cap = "".join(list(cap))

            return {"caption": cap, "model": self.model_version}

        finally:
            tmp_path.unlink(missing_ok=True)

    @modal.method()
    def ocr(self, s3_key: str) -> dict:
        from PIL import Image

        tmp_path = _download_image(s3_key)
        try:
            pil = Image.open(tmp_path).convert("RGB")
            out = self._model.query(
                image=pil,
                question="Transcribe the text in natural reading order.",
                stream=False,
            )
            text = out.get("answer", "") if isinstance(out, dict) else str(out)
            if not isinstance(text, str):
                text = "".join(list(text))
            return {"text": text, "model": self.model_version}
        finally:
            tmp_path.unlink(missing_ok=True)


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
    def load_model(self):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            dtype=torch.float16,
            device_map="auto",
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map="auto",
            dtype=torch.float16,
            attn_implementation="sdpa",
        )
        self._is_siglip2 = "siglip2" in self.model_name.lower()
        self._is_naflex = "naflex" in self.model_name.lower()
        self._has_image_features = hasattr(self.model, "get_image_features")
        self._has_text_features = hasattr(self.model, "get_text_features")

    def _to_device(self, inputs: dict) -> dict:
        import torch

        out = {}
        for k, v in inputs.items():
            if v.dtype == torch.float32:
                out[k] = v.to(device=self.model.device, dtype=self.model.dtype)
            else:
                out[k] = v.to(self.model.device)
        return out

    def _encode_images(self, images: list):
        import torch

        if self._is_naflex:
            inputs = self.processor(
                images=images, return_tensors="pt", padding="max_length", max_num_patches=256
            )
        else:
            inputs = self.processor(images=images, return_tensors="pt", padding="max_length")

        inputs = self._to_device(inputs)

        with torch.no_grad():
            if self._has_image_features:
                feats = self.model.get_image_features(**inputs)
            else:
                out = self.model(**inputs)
                if hasattr(out, "image_embeds"):
                    feats = out.image_embeds
                elif hasattr(out, "last_hidden_state"):
                    feats = out.last_hidden_state[:, 0, :]
                else:
                    raise ValueError("Could not extract image features")
        return feats.cpu().numpy()

    def _encode_texts(self, texts: list[str]):
        import torch

        if self._is_siglip2:
            texts = [t.lower() for t in texts]
            inputs = self.processor(
                text=texts, return_tensors="pt", padding="max_length", max_length=64
            )
        else:
            inputs = self.processor(text=texts, return_tensors="pt", padding="max_length")

        inputs = self._to_device(inputs)

        with torch.no_grad():
            if self._has_text_features:
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
                    raise ValueError("Unknown text output format")
        return feats.cpu().numpy()

    @modal.method()
    def embed_batch(self, items: list[dict]) -> dict:
        from PIL import Image

        results = []
        failed_ids = []

        for item in items:
            try:
                tmp_path = _download_image(item["s3_key"])
                try:
                    pil = Image.open(tmp_path)
                    if pil.mode == "P" and "transparency" in pil.info:
                        pil = pil.convert("RGBA")
                    pil = pil.convert("RGB")

                    img_feats = self._encode_images([pil])
                    txt_feats = self._encode_texts([item.get("text", "")])

                    model_slug = self.model_name.replace("/", "_")
                    source = item.get("dataset") or "api-ingested"
                    sha = item["sha256"]

                    img_key = f"embeddings/{model_slug}/{source}/{sha}.npy"
                    txt_key = f"embeddings/{model_slug}/{source}/{sha}_text.npy"

                    _upload_numpy(img_feats[0], img_key)
                    _upload_numpy(txt_feats[0], txt_key)

                    results.append(
                        {
                            "image_id": item["image_id"],
                            "image_embedding_key": img_key,
                            "text_embedding_key": txt_key,
                            "model": self.model_name,
                            "dimension": int(img_feats.shape[-1]),
                        }
                    )
                finally:
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                failed_ids.append(item["image_id"])

        return {"results": results, "failed_ids": failed_ids}

    @modal.method()
    def encode_text(self, query: str) -> dict:
        feats = self._encode_texts([query])
        return {
            "embedding": feats[0].tolist(),
            "model": self.model_name,
            "dimension": int(feats.shape[-1]),
        }
