

from __future__ import annotations
from typing import Protocol, runtime_checkable, Optional
from PIL import Image as PILImage

from ..schemas import OCRResult, DetectionResult, CaptionResult

@runtime_checkable
class VisionModel(Protocol):
    name: str
    def caption(self, image: PILImage.Image, length: str = "short") -> CaptionResult: ...
    def ocr(self, image: PILImage.Image) -> OCRResult: ...
    def detect(self, image: PILImage.Image) -> DetectionResult: ...

_REGISTRY: dict[str, type[VisionModel]] = {}

def register(name: str):
    def _wrap(cls):
        _REGISTRY[name] = cls
        return cls
    
    return _wrap

def create_vision_model(name: str, **kwargs) -> VisionModel:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown vision model: {name}. Available models: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](**kwargs)