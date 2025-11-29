from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    id: int = Field(..., description="Image Id")
    sha256: str = Field(..., description="Image hash")
    score: float = Field(..., description="Similarity Score")
    url: str | None = Field(None, description="Image url")
    rel_path: str = Field(..., description="Relative path to img")
    caption: str | None = Field(None, description="Generated Caption")
    ocr_text: str | None = Field(None, description="Extracted text from the image")
    width: int | None = Field(None, description="Image width")
    height: int | None = Field(None, description="Image height")
