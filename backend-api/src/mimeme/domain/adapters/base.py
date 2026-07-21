from http import HTTPMethod
from random import Random
from typing import Any, Protocol

from pydantic import BaseModel, Field


class FetchRequest(BaseModel, frozen=True):
    url: str
    method: HTTPMethod = HTTPMethod.GET
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = 30.0


class DiscoveredItem(BaseModel, frozen=True):
    external_item_id: str
    media_url: str
    canonical_item_url: str | None = None
    canonical_image_url: str | None = None
    title: str | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class Adapter(Protocol):
    key: str

    def build_requests(self, config: dict[str, Any], *, rng: Random) -> list[FetchRequest]: ...

    def parse(self, raw: dict[str, Any]) -> list[DiscoveredItem]: ...
