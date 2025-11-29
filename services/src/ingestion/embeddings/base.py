from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageFile

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class EmbedderConfig:
    image_model: str = "google/siglip2-base-patch16-naflex"
    # if none -> uses the same cross modal model's text encoder
    text_model: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_bnb_4bit: bool = False
    fp16_fallback: bool = True
    batch_size = 8


class BaseEmbedder(ABC):
    def __init__(self, cfg: EmbedderConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.image_model_name = cfg.image_model
        self.text_model_name = cfg.text_model

        self._load_models()

    @abstractmethod
    def _load_models(self):
        pass

    @abstractmethod
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        pass

    @abstractmethod
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        pass

    def embed_batch(
        self, image_paths: list[tuple[int, str]], text_provider: Callable
    ) -> list[tuple[int, np.ndarray, np.ndarray]]:
        results: list[tuple[int, np.ndarray, np.ndarray]] = []

        batch_size = max(1, min(self.cfg.batch_size, len(image_paths)))

        for i in range(0, len(image_paths), batch_size):
            chunk = image_paths[i : i + batch_size]
            pil_images = []
            texts = []
            valid_items = []

            for img_id, img_path in chunk:
                try:
                    pil = Image.open(img_path)

                    if pil.mode == "P" and "transparency" in pil.info:
                        pil = pil.convert("RGBA")

                    pil = pil.convert("RGB")

                    text = text_provider(img_id)

                    pil_images.append(pil)
                    texts.append(text or "")
                    valid_items.append((img_id, img_path))
                except Exception as e:
                    print(f"Failed to load image: {img_path}: {e}")
                    continue
            if not pil_images:
                break

            try:
                img_feats = self.encode_images(pil_images)
            except Exception as e:
                print(f"\nFATAL: Failed to encode images (batch size {len(pil_images)})")
                print(f"  Image IDs: {[img_id for img_id, _ in valid_items]}")
                print(f"  Error: {e}")
                import traceback

                traceback.print_exc()
                raise RuntimeError(
                    f"Image encoding failed for batch with IDs {[img_id for img_id, _ in valid_items]}"
                ) from e

            try:
                txt_feats = self.encode_texts(texts)
            except Exception as e:
                print(f"\nFATAL: Failed to encode texts (batch size {len(texts)})")
                print(f"  Image IDs: {[img_id for img_id, _ in valid_items]}")
                print(f"  Error: {e}")
                import traceback

                traceback.print_exc()
                raise RuntimeError(
                    f"Text encoding failed for batch with IDs {[img_id for img_id, _ in valid_items]}"
                ) from e

            for (img_id, _), imf, tf in zip(valid_items, img_feats, txt_feats):
                results.append((img_id, imf, tf))

        return results
