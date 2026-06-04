from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Protocol, cast

import structlog
from PIL.Image import Image
from transformers import AutoModelForCausalLM

from activities.vision.models import AnnotateImageOutput, VisionModelConfig

if TYPE_CHECKING:

    class MoondreamModel(Protocol):
        def encode_image(self, image: Image) -> object: ...

        def caption(
            self, image: object, length: str = "normal", stream: bool = False
        ) -> dict[str, str]: ...

        def query(
            self, image: object, question: str, reasoning: bool = True, stream: bool = False
        ) -> dict[str, str]: ...


class Moondream2:
    # only a single moondream instance per process
    _instance: Moondream2 | None = None
    _lock = threading.Lock()
    _ocr_prompt = "Transcribe the text in natural reading order."
    _log = structlog.get_logger().bind(component="moondream2")

    def __init__(self, config: VisionModelConfig) -> None:
        started = time.monotonic()
        outcome = "success"
        error_type: str | None = None
        error_message: str | None = None
        self.config = config
        log = self._log.bind(
            model_id=config.model_id, model_revision=config.revision, model_device=config.device
        )
        log.info("vision_step", step="model_load_start")
        try:
            loaded_model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                revision=config.revision,
                trust_remote_code=True,
                device_map={"": config.device},
            )
            self.model = cast("MoondreamModel", loaded_model)
            log.info("vision_step", step="model_load_complete")

            if config.compile_model:
                inner = getattr(loaded_model, "model", None)
                if inner is not None and hasattr(inner, "compile"):
                    log.info("vision_step", step="model_compile_start")
                    inner.compile()
                    log.info("vision_step", step="model_compile_complete")
        except Exception as exc:
            outcome = "error"
            error_type = type(exc).__name__
            error_message = str(exc)
            log.error(
                "vision_step",
                step="model_init_failed",
                error_type=error_type,
                error=error_message,
                exc_info=True,
            )
            raise
        finally:
            log.info(
                "vision_wide_event",
                event_type="vision_wide_event",
                phase="model_init",
                outcome=outcome,
                duration_ms=int((time.monotonic() - started) * 1000),
                error_type=error_type,
                error=error_message,
                compile_model=config.compile_model,
            )

    @classmethod
    def get_instance(cls, config: VisionModelConfig | None = None) -> Moondream2:
        if cls._instance is None:
            cls._log.info("vision_step", step="singleton_cache_miss")
            with cls._lock:
                if cls._instance is None:
                    cfg = config or VisionModelConfig()
                    cls._instance = cls(cfg)
                    cls._log.info("vision_step", step="singleton_instance_created")
        else:
            cls._log.info("vision_step", step="singleton_cache_hit")
        # double checked locking ensures this
        assert cls._instance is not None
        return cls._instance

    @property
    def model_version(self) -> str:
        return f"{self.config.model_id}@{self.config.revision or 'latest'}"

    def annotate_image(self, image: Image, length: str = "normal") -> AnnotateImageOutput:
        encoded_image = self.model.encode_image(image)
        caption = self.model.caption(encoded_image, length=length)["caption"]
        ocr_text = self.model.query(encoded_image, self._ocr_prompt, reasoning=False)["answer"]
        return AnnotateImageOutput(
            image_id=0,
            caption=caption,
            caption_model=self.model_version,
            ocr_text=ocr_text,
            ocr_model=self.model_version,
        )
