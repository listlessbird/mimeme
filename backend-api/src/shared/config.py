from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="development")
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    db_url: PostgresDsn = cast(PostgresDsn, "postgresql://postgres:postgres@localhost:5432/mimeme")
    redis_url: RedisDsn = cast(RedisDsn, "redis://localhost:6379/0")

    api_key_admin: str | None = None
    api_key_readonly: str | None = None

    temporal_host: str = Field(default="localhost:7233")
    temporal_namespace: str = Field(default="default")
    temporal_task_queue: str = Field(default="findmeme-tasks")

    s3_endpoint_url: str = Field(default="http://localhost:9000")
    s3_region: str = Field(default="us-east-1")
    s3_access_key_id: str = Field(default="minioadmin")
    s3_secret_access_key: str = Field(default="minioadmin")
    s3_bucket: str = Field(default="mimeme")
    s3_force_path_style: bool = Field(default=True)
    s3_presigned_url_expiry: int = Field(default=3600)

    vision_model: str = Field(default="vikhyatk/moondream2")
    vision_model_revision: str | None = Field(default="2025-06-21")
    embed_model: str = Field(default="google/siglip2-base-patch16-naflex")
    embed_device: str = Field(default="cuda")

    index_type: str = Field(default="flat")
    faiss_hnsw_ef_search: int = Field(default=128)
    index_cache_dir: Path = Field(default=Path("data/cache/indexes"))

    # modal integration in prod
    gpu_backend: Literal["local", "modal"] = Field(default="local")
    modal_app_name: str = Field(default="findmeme-gpu")

    onnx_text_encoder_repo: str = Field(
        default="listlessbird/siglip2-base-patch16-naflex-text-onnx"
    )
    onnx_text_encoder_revision: str = Field(default="092dc08370b1a01d69c78067051b124881a95407")
    onnx_text_encoder_variant: str = Field(default="text_model_int8.onnx")
    onnx_text_encoder_threads: int = Field(default=4)
    preload_text_encoder_on_startup: bool = Field(default=True)

    axiom_api_token: str = ""
    axiom_dataset: str = ""

    @field_validator("index_cache_dir", mode="before")
    @classmethod
    def parse_path(cls, v: str | Path) -> Path:
        return Path(v)

    @property
    def db_url_str(self) -> str:
        # SQLAlchemy expects the dialect name "postgresql", not "postgres".
        db_url = str(self.db_url)
        if db_url.startswith("postgres://"):
            return "postgresql://" + db_url[len("postgres://") :]
        return db_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()
