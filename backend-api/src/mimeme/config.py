from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import Field, PostgresDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

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
    task_queue: str = "mimeme-v2"


class HttpConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HTTP_", env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    request_timeout_s: float = 30.0
    loop_lag_threshold_ms: float = 50.0
    ready_cache_s: float = 5.0
    rate_limit_enabled: bool = True
    cors_origins: list[str] = ["https://mimeme.dev"]
    api_key_admin: SecretStr | None = None
    api_key_readonly: SecretStr | None = None


class AuthConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AUTH_", env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    github_client_id: str | None = None
    github_client_secret: SecretStr | None = None
    github_callback_url: str = "http://localhost:8000/auth/github/callback"
    allowed_github_ids: Annotated[frozenset[str], NoDecode] = frozenset()
    session_secret: SecretStr | None = None
    session_cookie: str = "mimeme_admin_session"
    session_max_age_s: int = 60 * 60 * 24 * 7
    cookie_domain: str | None = None
    ui_url: str = "http://localhost:3000"

    @field_validator("allowed_github_ids", mode="before")
    @classmethod
    def parse_allowed_github_ids(cls, value: object) -> object:
        if isinstance(value, str):
            return frozenset(item.strip() for item in value.split(",") if item.strip())
        return value


class ComputeConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="COMPUTE_", env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    gpu_backend: Literal["local", "modal"] = "local"
    modal_app_name: str = "findmeme-gpu"
    modal_hf_cache_volume_name: str = "findmeme-hf-cache"
    modal_s3_secret_name: str = "findmeme-s3"

    gateway_url: str = "http://127.0.0.1:8010"
    inference_gateway_url: str | None = None
    bind_host: str = "0.0.0.0"
    bind_port: int = 8010
    socket_dir: Path = Path("/tmp/mimeme-compute")
    request_timeout_s: float = 60.0
    poll_interval_s: float = 5.0
    heartbeat_timeout_s: float = 30.0
    child_grace_s: float = 5.0
    job_io_concurrency: int = Field(default=4, ge=1)

    @field_validator("socket_dir", mode="before")
    @classmethod
    def _parse_socket_dir(cls, v: str | Path) -> Path:
        return Path(v)


class InferenceConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_", env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    vision_model: str = "vikhyatk/moondream2"
    vision_model_revision: str | None = "2025-06-21"
    vision_compile: bool = False
    embed_model: str = "google/siglip2-base-patch16-naflex"
    embed_device: str = "cuda"
    residency: Literal["both", "swap"] = "both"
    embed_batch_size: int = Field(default=4, ge=1)


class SearchConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SEARCH_", env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    encoder_repo: str = "listlessbird/siglip2-base-patch16-naflex-text-onnx"
    encoder_revision: str = "092dc08370b1a01d69c78067051b124881a95407"
    encoder_variant: str = "text_model_int8.onnx"
    encoder_threads: int = 4
    hnsw_ef_search: int = 128


class IndexConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INDEX_", env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    type: Literal["flat", "hnsw"] = "flat"
    faiss_hnsw_ef_search: int = 128
    cache_dir: Path = Path("data/cache/indexes")

    rebuild_claim_timeout_minutes: int = 180
    rebuild_schedule_enabled: bool = True
    rebuild_schedule_cron: str | None = "*/5 * * * *"
    rebuild_schedule_timezone: str = "UTC"
    rebuild_settle_minutes: int = 10
    rebuild_max_stale_hours: int = 6
    reconcile_interval_s: float = 30.0
    retain_versions: int = 5
    build_threads: int = 2
    shard_rows: int = 1_500
    seal_max_shards: int = 50
    seal_min_rows: int = 100

    @field_validator("cache_dir", mode="before")
    @classmethod
    def parse_path(cls, v: str | Path) -> Path:
        return Path(v)

    @field_validator("rebuild_schedule_cron", mode="before")
    @classmethod
    def parse_optional_cron(cls, value: object) -> object:
        return None if value == "" else value


class LogConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOG_", env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    level: str = "INFO"
    axiom_api_token: SecretStr = SecretStr("")
    axiom_query_token: SecretStr = SecretStr("")
    axiom_dataset: str = ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    # Defaults to False so a deployment that forgets to set DEBUG fails
    # closed (docs/openapi/CORS stay locked down). Development .env files set
    # DEBUG=true explicitly.
    debug: bool = False
    tumblr_api_key: SecretStr | None = None

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    http: HttpConfig = Field(default_factory=HttpConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    logging: LogConfig = Field(default_factory=LogConfig)
