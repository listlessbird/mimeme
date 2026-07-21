from __future__ import annotations

from pathlib import Path
from typing import Literal, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import Field, PostgresDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = ".env"


class DatabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DATABASE_", env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    url: PostgresDsn = cast(PostgresDsn, "postgresql://postgres:postgres@localhost:5432/mimeme")
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout_s: float = 5.0
    statement_cache_size: int = 100

    @property
    def url_str(self) -> str:
        # SQLAlchemy expects the dialect name "postgresql", not "postgres".
        url = str(self.url)
        if url.startswith("postgres://"):
            return "postgresql://" + url[len("postgres://") :]
        return url

    @property
    def async_url_str(self) -> str:
        parts = urlsplit(self.url_str)
        query_items = [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if key not in {"sslmode", "channel_binding"}
        ]
        return urlunsplit(
            (
                "postgresql+asyncpg",
                parts.netloc,
                parts.path,
                urlencode(query_items),
                parts.fragment,
            )
        )

    @property
    def ssl_required(self) -> bool:
        parts = urlsplit(self.url_str)
        params = dict(parse_qsl(parts.query, keep_blank_values=True))
        return params.get("sslmode") in {"require", "verify-ca", "verify-full"}


class MediaConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEDIA_", env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    s3_endpoint_url: str = "http://localhost:9000"
    s3_region: str = "us-east-1"
    s3_access_key_id: str = "minioadmin"
    s3_secret_access_key: SecretStr = SecretStr("minioadmin")
    s3_bucket: str = "mimeme-media"
    s3_force_path_style: bool = True
    public_base_url: str = "http://localhost:9000/mimeme-media"


class ArtifactConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARTIFACT_", env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    s3_endpoint_url: str = "http://localhost:9000"
    s3_region: str = "us-east-1"
    s3_access_key_id: str = "minioadmin"
    s3_secret_access_key: SecretStr = SecretStr("minioadmin")
    s3_bucket: str = "mimeme-artifacts"
    s3_force_path_style: bool = True


class TemporalConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TEMPORAL_", env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    host: str = "localhost:7233"
    namespace: str = "default"
    task_queue: str = "findmeme-tasks"


class HttpConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HTTP_", env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    request_timeout_s: float = 30.0
    loop_lag_threshold_ms: float = 50.0
    rate_limit_enabled: bool = True
    api_key_admin: SecretStr | None = None
    api_key_readonly: SecretStr | None = None


class ComputeConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="COMPUTE_", env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    gpu_backend: Literal["local", "modal"] = "local"
    modal_app_name: str = "findmeme-gpu"
    modal_hf_cache_volume_name: str = "findmeme-hf-cache"
    modal_s3_secret_name: str = "findmeme-s3"


class InferenceConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_", env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    vision_model: str = "vikhyatk/moondream2"
    vision_model_revision: str | None = "2025-06-21"
    embed_model: str = "google/siglip2-base-patch16-naflex"
    embed_device: str = "cuda"

    onnx_text_encoder_repo: str = "listlessbird/siglip2-base-patch16-naflex-text-onnx"
    onnx_text_encoder_revision: str = "092dc08370b1a01d69c78067051b124881a95407"
    onnx_text_encoder_variant: str = "text_model_int8.onnx"
    onnx_text_encoder_threads: int = 4
    preload_text_encoder_on_startup: bool = True


class IndexConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INDEX_", env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    type: str = "flat"
    faiss_hnsw_ef_search: int = 128
    cache_dir: Path = Path("data/cache/indexes")

    rebuild_claim_timeout_minutes: int = 180
    rebuild_schedule_enabled: bool = True
    rebuild_schedule_cron: str = "* * * * *"
    rebuild_schedule_timezone: str = "UTC"

    @field_validator("cache_dir", mode="before")
    @classmethod
    def parse_path(cls, v: str | Path) -> Path:
        return Path(v)


class LogConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOG_", env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    level: str = "INFO"
    axiom_api_token: SecretStr = SecretStr("")
    axiom_dataset: str = ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    debug: bool = True

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    http: HttpConfig = Field(default_factory=HttpConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    logging: LogConfig = Field(default_factory=LogConfig)
