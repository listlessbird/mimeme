from __future__ import annotations

from pydantic import BaseModel, Field


class VisionModelConfig(BaseModel):
    model_id: str = Field(default="vikhyatk/moondream2")
    revision: str | None = Field(default="2025-06-21")
    device: str = Field(default="cuda")
    compile_model: bool = Field(default=True)

    model_config = {"frozen": True}


class AnnotateImageInput(BaseModel):
    image_id: int
    s3_key: str
    length: str = Field(default="normal")


class AnnotateImageOutput(BaseModel):
    image_id: int
    caption: str
    caption_model: str
    ocr_text: str
    ocr_model: str
