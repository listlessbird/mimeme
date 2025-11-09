

from pydantic import BaseModel, Field
from typing import Optional, Literal, List

status = Literal["pending", "running", "done", "failed"]

class CaptionResult(BaseModel):
    caption: str
    model: str
    tokens: Optional[int] = None

class OCRResult(BaseModel):
    text: str
    model: str
    tokens: Optional[int] = None

class DetectionBox(BaseModel):
    label: str
    score: float
    box: tuple[int, int, int, int] # x1, y1, x2, y2

class DetectionResult(BaseModel):
    objects: List[DetectionBox] = Field(default_factory=list)
    model: str