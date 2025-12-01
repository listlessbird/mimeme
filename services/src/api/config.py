from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "find-meme"
    app_env: Literal["development", "production"] = "production"
    debug: bool = Field(default=True)
    log_level: str = "INFO"

    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)

    @computed_field
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @computed_field
    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @computed_field
    @property
    def index_cache_dir(self) -> Path:
        return self.cache_dir / "indexes"

    database_url: str | None = None

    @computed_field
    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        return "postgresql://findmeme:findmeme@localhost:5432/findmeme"

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None

    @computed_field
    @property
    def broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @computed_field
    @property
    def result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url

    s3_endpoint_url: str = "http://localhost:9000"
    s3_region: str = "auto"
    s3_bucket: str = "findmeme"
    s3_access_key_id: str = "minioadmin"
    s3_secret_access_key: str = "minioadmin"
    s3_force_path_style: bool = True
    s3_presigned_url_expiry: int = 3600

    embed_model: str = "google/siglip2-base-patch16-naflex"
    embed_device: str = "cuda"
    embed_batch_size: int = 8
    vision_model: str = "moondream2"

    index_type: Literal["flat", "ivf", "hnsw"] = "flat"
    index_nlist: int = 100  # for IVF
    index_nprobe: int = 10  # for IVF search
    index_retain_versions: int = 5  # keep n old indexes

    search_default_limit: int = 20
    search_max_limit: int = 50

    worker_concurrency: int = 2  # celery workers
    ingest_batch_size: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
