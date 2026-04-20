import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_snake

from shared.models.orm import SourceRunStatus, SourceRunTrigger


class SourceItemData(BaseModel):
    external_item_id: str
    canonical_item_url: str | None = None
    fetch_url: str
    canonical_media_id: str
    media_type: str
    title: str
    published_at: datetime.datetime | None = None
    raw_metadata: dict[str, Any] | None = None


class FetchSourceItemsOutput(BaseModel):
    items: list[SourceItemData]
    # out might have gifs which we dont support rn
    skipped: int = 0


class SourceConfig(BaseModel):
    source_id: int
    adapter_key: str
    adapter_config: dict[str, Any]
    secret_refs: list[str] | None = None
    max_items_per_run: int = 50
    # annotes img under this name in object storage
    dataset: str | None = None
    default_tags: dict[str, Any] | None = None


class CreateSourceRunInput(BaseModel):
    source_id: int
    trigger_mode: SourceRunTrigger


class CreateSourceRunOutput(BaseModel):
    source_run_id: int


class CompleteSourceRunInput(BaseModel):
    source_run_id: int
    source_id: int
    status: SourceRunStatus
    error_message: str | None = None
    summary: dict[str, Any] | None = None


class FetchSourceItemsInput(BaseModel):
    source_id: int
    adapter_key: str
    adapter_config: dict[str, Any]
    secret_refs: list[str] | None = None
    max_items_per_run: int = 50


class FilterSeenItemsInput(BaseModel):
    source_id: int
    items: list[SourceItemData]


class FilterSeenItemsOutput(BaseModel):
    new_items: list[SourceItemData]
    seen_count: int


class PersistSourceItemsInput(BaseModel):
    source_id: int
    source_run_id: int
    items: list[SourceItemData]


class PersistSourceItemsOutput(BaseModel):
    source_item_ids: list[int]


class CreateSourceIngestInput(BaseModel):
    source_id: int
    source_run_id: int
    job_id: str
    items: list[SourceItemData]
    source_item_ids: list[int]
    dataset: str | None = None


class ApiLeagueMeme(BaseModel):
    description: str
    url: str
    type: str
    width: int | None = None
    height: int | None = None
    ratio: float | None = None


class MemeApiMeme(BaseModel):
    model_config = ConfigDict(alias_generator=to_snake, populate_by_name=True)

    post_link: str
    subreddit: str
    title: str
    url: str
    nsfw: bool
    spoiler: bool
    author: str
    ups: int
    preview: list[str]


class MemeApiResponse(BaseModel):
    count: int
    memes: list[MemeApiMeme]
