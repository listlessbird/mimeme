from __future__ import annotations

import re
from random import Random
from typing import Any, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from mimeme.source.model import DiscoveredItem, FetchRequest

MEME_API_URL = "https://meme-api.com/gimme"
MAX_PER_CALL = 50

_BLOCKED_MEDIA_EXT = frozenset({".gif", ".gifv", ".mp4", ".webm", ".mov", ".m4v"})

_REDDIT_POST_ID_RE = re.compile(
    r"(?:redd\.it/|reddit\.com/r/[^/]+/comments/)([a-z0-9]+)",
    re.IGNORECASE,
)

_RAW_METADATA_KEYS = ("author", "title", "ups", "subreddit", "preview", "postLink")


class UnknownAdapterKey(Exception):
    pass


class Adapter(Protocol):
    key: str

    def build_requests(self, config: dict[str, Any], *, rng: Random) -> list[FetchRequest]: ...

    def parse(self, raw: dict[str, Any]) -> list[DiscoveredItem]: ...


class MemeApiConfig(BaseModel):
    subreddits: list[str] = Field(min_length=1)
    max_items_per_run: int | None = Field(default=None, ge=0)


def parse_external_item_id(post_link: str | None) -> str | None:
    if not post_link:
        return None
    match = _REDDIT_POST_ID_RE.search(post_link)
    if match is None:
        return None
    return match.group(1).lower()


def is_still_image_url(url: str | None) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    if parsed.netloc.lower() == "v.redd.it":
        return False
    path = parsed.path.lower()
    return not any(path.endswith(ext) for ext in _BLOCKED_MEDIA_EXT)


class MemeApiAdapter:
    key = "meme_api"

    def build_requests(self, config: dict[str, Any], *, rng: Random) -> list[FetchRequest]:
        cfg = MemeApiConfig.model_validate(config)

        subreddits = list(cfg.subreddits)
        rng.shuffle(subreddits)

        remaining = (
            cfg.max_items_per_run
            if cfg.max_items_per_run is not None
            else len(subreddits) * MAX_PER_CALL
        )

        requests: list[FetchRequest] = []
        for subreddit in subreddits:
            if remaining <= 0:
                break
            count = min(MAX_PER_CALL, remaining)
            requests.append(FetchRequest(url=f"{MEME_API_URL}/{subreddit}/{count}"))
            remaining -= count

        return requests

    def parse(self, raw: dict[str, Any]) -> list[DiscoveredItem]:
        raw_items = raw.get("memes")
        memes = raw_items if isinstance(raw_items, list) else [raw]

        items: list[DiscoveredItem] = []
        for meme in memes:
            if not isinstance(meme, dict):
                continue

            external_item_id = parse_external_item_id(meme.get("postLink"))
            if external_item_id is None:
                continue

            if meme.get("nsfw"):
                continue

            media_url = meme.get("url")
            if not isinstance(media_url, str):
                continue

            if not is_still_image_url(media_url):
                continue

            items.append(
                DiscoveredItem(
                    external_item_id=external_item_id,
                    media_url=media_url,
                    canonical_item_url=meme.get("postLink"),
                    title=meme.get("title"),
                    raw_metadata={key: meme.get(key) for key in _RAW_METADATA_KEYS},
                )
            )

        return items


ADAPTERS: dict[str, Adapter] = {adapter.key: adapter for adapter in (MemeApiAdapter(),)}

KNOWN_ADAPTER_KEYS = frozenset(ADAPTERS)


def get_adapter(key: str) -> Adapter:
    try:
        return ADAPTERS[key]
    except KeyError:
        raise UnknownAdapterKey(key) from None
