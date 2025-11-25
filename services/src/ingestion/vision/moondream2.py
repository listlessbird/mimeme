from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from PIL import Image as PILImage

import torch
from transformers import AutoModelForCausalLM
from ..schemas import CaptionResult, OCRResult, DetectionResult
from .base import VisionModel, register

@dataclass
class MoonDream2Config:
    hf_id: str = "vikhyatk/moondream2"
    revision: Optional[str] = "2025-06-21"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile_model: bool = False
    prompt_ocr: str = "Transcribe the text in natural reading order."

@register("moondream2")
class MoonDream2(VisionModel):
    name = "moondream2"

    def __init__(self, cfg: Optional[MoonDream2Config] = None):
        self.cfg = cfg or MoonDream2Config()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.hf_id,
            revision=self.cfg.revision,
            trust_remote_code=True,
            device_map={"": self.cfg.device},
        )

        if self.cfg.compile_model and hasattr(self.model, "model"):
            try:
                self.model.model.compile()
            except Exception as e:
                print(f"Failed to compile model: {e}")
                pass

    def caption(self, image: PILImage.Image, length: str = "short") -> CaptionResult:
        out = self.model.caption(image, length=length)
        cap = out["caption"] if isinstance(out, dict) else str(out)
        return CaptionResult(caption=cap, model=f"{self.cfg.hf_id}@{self.cfg.revision or 'latest'}")

    def ocr(self, image: PILImage.Image) -> OCRResult:
        out = self.model.query(image, self.cfg.prompt_ocr)
        txt = out.get("answer", "") if isinstance(out, dict) else str(out)
        return OCRResult(text=txt, model=f"{self.cfg.hf_id}@{self.cfg.revision or 'latest'}")

    def detect(self, image: PILImage.Image) -> DetectionResult:
        res = self.model.detect(image)
        objects = []
        for o in res.get("objects", []):
            objects.append({
                "label": o.get("label", ""),
                "score": o.get("score", 0.0),
                "box": tuple(o.get("box", (0, 0, 0, 0))),
            })
        return DetectionResult(objects=objects, model=f"{self.cfg.hf_id}@{self.cfg.revision or 'latest'}")
        