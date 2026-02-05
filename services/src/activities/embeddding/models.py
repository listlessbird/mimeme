from pydantic import BaseModel, Field


class EmbedderConfig(BaseModel):
    image_model: str = Field(default="google/siglip2-base-patch16-naflex")
    text_model: str | None = Field(default=None)
    device: str = Field(default="cuda")
    use_bnb_4bit: bool = Field(default=False)
    fp16_fallback: bool = Field(default=True)
    batch_size: int = Field(default=8)

    model_config = {"frozen": True}


class EmbedImageInput(BaseModel):
    image_id: int
    s3_key: str
    text: str = Field(default="")


class EmbedImageOutput(BaseModel):
    image_id: int
    image_embedding_key: str
    text_embedding_key: str
    model: str
    dimension: int


class EmbedBatchInput(BaseModel):
    items: list[EmbedImageInput]
    dataset: str | None = Field(default=None)


class EmbedBatchOutput(BaseModel):
    results: list[EmbedImageOutput]
    failed_ids: list[int]
