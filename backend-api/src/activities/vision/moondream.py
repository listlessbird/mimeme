from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Protocol

from PIL.Image import Image
from transformers import AutoModelForCausalLM

from activities.vision.models import CaptionOutput, OCROutput, VisionModelConfig

if TYPE_CHECKING:

    class MoondreamModel(Protocol):
        def caption(self, image: Image, length: str) -> dict[str, Any] | str: ...
        def query(
            self, image: Image, question: str, stream: bool = False
        ) -> dict[str, Any] | str: ...


class Moondream2:
    # only a single moondream instance per process
    _instance: Moondream2 | None = None
    _lock = threading.Lock()
    _ocr_prompt = "Transcribe the text in natural reading order."

    def __init__(self, config: VisionModelConfig) -> None:
        self.config = config
        loaded_model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=True,
            device_map={"": config.device},
        )
        self.model: MoondreamModel = loaded_model

        if config.compile_model:
            inner = getattr(loaded_model, "model", None)
            if inner is not None and hasattr(inner, "compile"):
                inner.compile()

    @classmethod
    def get_instance(cls, config: VisionModelConfig | None = None) -> Moondream2:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cfg = config or VisionModelConfig()
                    cls._instance = cls(cfg)
        # double checked locking ensures this
        assert cls._instance is not None
        return cls._instance

    @property
    def model_version(self) -> str:
        return f"{self.config.model_id}@{self.config.revision or 'latest'}"

    def caption(self, image: Image, length: str = "normal") -> CaptionOutput:
        out = self.model.caption(image, length)
        cap = out.get("caption", "") if isinstance(out, dict) else out

        if not isinstance(cap, str):
            cap = "".join(list(cap))

        # ill override teh image id at the caller
        return CaptionOutput(image_id=0, caption=cap, model=self.model_version)

    def ocr(self, image: Image, prompt: str = _ocr_prompt) -> OCROutput:
        out = self.model.query(image=image, question=prompt, stream=False)
        text = out.get("answer", "") if isinstance(out, dict) else str(out)

        if not isinstance(text, str):
            text = "".join(list(text))

        return OCROutput(image_id=0, text=text, model=self.model_version)
