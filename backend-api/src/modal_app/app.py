from __future__ import annotations

import io
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import boto3
import modal
import structlog
from botocore.config import Config as BotoConfig

app = modal.App("findmeme-gpu")

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
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
        "structlog>=25.5",
    )
)

hf_cache = modal.Volume.from_name("findmeme-hf-cache", create_if_missing=True)

HF_CACHE_DIR = "/root/.cache/huggingface"

s3_secret = modal.Secret.from_name("findmeme-s3")


def _setup_logging() -> structlog.BoundLogger:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger().bind(
        service="modal-gpu",
        modal_app_name="findmeme-gpu",
        app_env=os.environ.get("APP_ENV", "production"),
    )


log = _setup_logging()


def _runtime_context() -> dict[str, object]:
    fields: dict[str, object] = {}
    for key in ("MODAL_TASK_ID", "MODAL_CONTAINER_ID", "MODAL_FUNCTION_ID"):
        value = os.environ.get(key)
        if value:
            fields[key.lower()] = value
    return fields


def _emit_modal_event(
    *,
    operation: str,
    started_at: float,
    outcome: str,
    error: str | None = None,
    **fields: object,
) -> None:
    event: dict[str, object] = {
        "event_type": "modal_wide_event",
        "operation": operation,
        "outcome": outcome,
        "duration_ms": int((time.monotonic() - started_at) * 1000),
        **_runtime_context(),
    }
    event.update(fields)
    if error:
        event["error"] = error
    log.info("modal_wide_event", **event)


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
    model_version: str = "vikhyatk/moondream2@2025-06-21"

    @modal.enter()
    def load_model(self):
        started = time.monotonic()
        outcome = "success"
        error_message: str | None = None
        from transformers import AutoModelForCausalLM

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                revision="2025-06-21",
                trust_remote_code=True,
                device_map={"": "cuda"},
            )
            inner = getattr(self._model, "model", None)
            if inner and hasattr(inner, "compile"):
                inner.compile()
        except Exception as exc:
            outcome = "error"
            error_message = str(exc)
            log.error("modal_step", operation="vision_load_model", step="failed", exc_info=True)
            raise
        finally:
            _emit_modal_event(
                operation="vision_load_model",
                started_at=started,
                outcome=outcome,
                error=error_message,
                model_version=self.model_version,
            )

    @modal.method()
    def caption(self, s3_key: str, length: str = "normal") -> dict:
        started = time.monotonic()
        outcome = "success"
        error_message: str | None = None
        from PIL import Image

        tmp_path = _download_image(s3_key)
        try:
            pil = Image.open(tmp_path).convert("RGB")
            caption_fn = getattr(self._model, "caption", None)
            if not callable(caption_fn):
                raise ValueError("Moondream model does not expose callable caption()")
            out = caption_fn(pil, length)
            cap = out.get("caption", "") if isinstance(out, dict) else out

            if not isinstance(cap, str):
                cap = "".join(list(cap))

            return {"caption": cap, "model": self.model_version}
        except Exception as exc:
            outcome = "error"
            error_message = str(exc)
            log.error("modal_step", operation="caption", step="failed", s3_key=s3_key, exc_info=True)
            raise
        finally:
            _emit_modal_event(
                operation="caption",
                started_at=started,
                outcome=outcome,
                error=error_message,
                s3_key=s3_key,
                caption_length=length,
                model_version=self.model_version,
            )

            tmp_path.unlink(missing_ok=True)

    @modal.method()
    def ocr(self, s3_key: str) -> dict:
        started = time.monotonic()
        outcome = "success"
        error_message: str | None = None
        from PIL import Image

        tmp_path = _download_image(s3_key)
        try:
            pil = Image.open(tmp_path).convert("RGB")
            query_fn = getattr(self._model, "query", None)
            if not callable(query_fn):
                raise ValueError("Moondream model does not expose callable query()")
            out = query_fn(pil, "Transcribe the text in natural reading order.")
            text = out.get("answer", "") if isinstance(out, dict) else str(out)
            if not isinstance(text, str):
                text = "".join(list(text))
            return {"text": text, "model": self.model_version}
        except Exception as exc:
            outcome = "error"
            error_message = str(exc)
            log.error("modal_step", operation="ocr", step="failed", s3_key=s3_key, exc_info=True)
            raise
        finally:
            _emit_modal_event(
                operation="ocr",
                started_at=started,
                outcome=outcome,
                error=error_message,
                s3_key=s3_key,
                model_version=self.model_version,
            )
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
        started = time.monotonic()
        outcome = "success"
        error_message: str | None = None
        import torch
        from transformers import AutoModel, AutoProcessor

        try:
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
        except Exception as exc:
            outcome = "error"
            error_message = str(exc)
            log.error("modal_step", operation="embedding_load_model", step="failed", exc_info=True)
            raise
        finally:
            _emit_modal_event(
                operation="embedding_load_model",
                started_at=started,
                outcome=outcome,
                error=error_message,
                model_name=self.model_name,
            )

    def _tensor_to_numpy(self, value: Any, *, kind: str):
        # Some model APIs return a tensor, others return an output object with tensor fields.
        if hasattr(value, "cpu"):
            return value.cpu().numpy()

        candidates = (
            ("image_embeds", "pooler_output", "last_hidden_state")
            if kind == "image"
            else ("text_embeds", "pooler_output", "last_hidden_state")
        )

        for field in candidates:
            if hasattr(value, field):
                tensor = getattr(value, field)
                if field == "last_hidden_state":
                    tensor = tensor[:, 0, :]
                if hasattr(tensor, "cpu"):
                    return tensor.cpu().numpy()

        raise ValueError(f"Could not convert {kind} features to numpy from type {type(value)!r}")

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
        return self._tensor_to_numpy(feats, kind="image")

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
        return self._tensor_to_numpy(feats, kind="text")

    @modal.method()
    def embed_batch(self, items: list[dict]) -> dict:
        started = time.monotonic()
        outcome = "success"
        error_message: str | None = None
        from PIL import Image

        results = []
        failed_ids = []

        def _prepare_item(item: dict) -> tuple[dict, Any]:
            tmp_path = _download_image(item["s3_key"])
            try:
                with Image.open(tmp_path) as pil:
                    if pil.mode == "P" and "transparency" in pil.info:
                        pil = pil.convert("RGBA")
                    rgb = pil.convert("RGB")
                    return item, rgb.copy()
            finally:
                tmp_path.unlink(missing_ok=True)

        prepared: list[tuple[dict, Any]] = []
        with ThreadPoolExecutor(max_workers=min(8, len(items) or 1)) as executor:
            future_to_item = {executor.submit(_prepare_item, item): item for item in items}
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    prepared.append(future.result())
                except Exception as exc:
                    failed_ids.append(item["image_id"])
                    log.error(
                        "modal_step",
                        operation="embed_batch",
                        step="item_prepare_failed",
                        image_id=item.get("image_id"),
                        s3_key=item.get("s3_key"),
                        error=str(exc),
                    )

        if prepared:
            ordered_items = [item for item, _ in prepared]
            images = [image for _, image in prepared]
            texts = [item.get("text", "") for item in ordered_items]

            img_feats = self._encode_images(images)
            txt_feats = self._encode_texts(texts)
            model_slug = self.model_name.replace("/", "_")
            dimension = int(img_feats.shape[-1])

            def _upload_one(position: int) -> dict:
                item = ordered_items[position]
                source = item.get("dataset") or "api-ingested"
                sha = item["sha256"]

                img_key = f"embeddings/{model_slug}/{source}/{sha}.npy"
                txt_key = f"embeddings/{model_slug}/{source}/{sha}_text.npy"

                _upload_numpy(img_feats[position], img_key)
                _upload_numpy(txt_feats[position], txt_key)

                return {
                    "image_id": item["image_id"],
                    "image_embedding_key": img_key,
                    "text_embedding_key": txt_key,
                    "model": self.model_name,
                    "dimension": dimension,
                }

            with ThreadPoolExecutor(max_workers=min(8, len(ordered_items))) as executor:
                future_to_item = {
                    executor.submit(_upload_one, position): ordered_items[position]
                    for position in range(len(ordered_items))
                }
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        failed_ids.append(item["image_id"])
                        log.error(
                            "modal_step",
                            operation="embed_batch",
                            step="item_upload_failed",
                            image_id=item.get("image_id"),
                            s3_key=item.get("s3_key"),
                            error=str(exc),
                        )

        if results:
            results.sort(key=lambda result: int(result["image_id"]))

        if failed_ids:
            outcome = "partial_failure" if results else "failed"
            error_message = f"{len(failed_ids)} items failed"

        payload = {"results": results, "failed_ids": failed_ids}
        _emit_modal_event(
            operation="embed_batch",
            started_at=started,
            outcome=outcome,
            error=error_message,
            item_count=len(items),
            processed=len(results),
            failed=len(failed_ids),
            model_name=self.model_name,
        )
        return payload

    @modal.method()
    def encode_text(self, query: str) -> dict:
        started = time.monotonic()
        outcome = "success"
        error_message: str | None = None
        dimension: int | None = None
        try:
            feats = self._encode_texts([query])
            dimension = int(feats.shape[-1])
            return {
                "embedding": feats[0].tolist(),
                "model": self.model_name,
                "dimension": dimension,
            }
        except Exception as exc:
            outcome = "error"
            error_message = str(exc)
            log.error("modal_step", operation="encode_text", step="failed", exc_info=True)
            raise
        finally:
            _emit_modal_event(
                operation="encode_text",
                started_at=started,
                outcome=outcome,
                error=error_message,
                query_chars=len(query),
                dimension=dimension,
                model_name=self.model_name,
            )
