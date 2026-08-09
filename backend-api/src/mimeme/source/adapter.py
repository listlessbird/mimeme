from __future__ import annotations

import re
from random import Random
from typing import Any, Protocol
from urllib.parse import urlencode, urlparse

from pydantic import BaseModel, Field

from mimeme.source.model import DiscoveredItem, FetchRequest

MEME_API_URL = "https://meme-api.com/gimme"
MAX_PER_CALL = 50
TUMBLR_TAGGED_URL = "https://api.tumblr.com/v2/tagged"
TUMBLR_MAX_PER_CALL = 20
TUMBLR_USER_AGENT = "MiMeMe/1.0 (https://mimeme.dev)"

_BLOCKED_MEDIA_EXT = frozenset({".gif", ".gifv", ".mp4", ".webm", ".mov", ".m4v"})

_REDDIT_POST_ID_RE = re.compile(
    r"(?:redd\.it/|reddit\.com/r/[^/]+/comments/)([a-z0-9]+)",
    re.IGNORECASE,
)

_RAW_METADATA_KEYS = ("author", "title", "ups", "subreddit", "preview", "postLink")
_TUMBLR_RAW_METADATA_KEYS = (
    "blog_name",
    "id_string",
    "post_url",
    "note_count",
    "tags",
    "type",
    "date",
    "caption",
    "source_url",
)


class UnknownAdapterKey(Exception):
    pass


class Adapter(Protocol):
    key: str

    def build_requests(self, config: dict[str, Any], *, rng: Random) -> list[FetchRequest]: ...

    def parse(self, raw: dict[str, Any], config: dict[str, Any]) -> list[DiscoveredItem]: ...


class MemeApiConfig(BaseModel):
    subreddits: list[str] = Field(min_length=1)
    min_score: int = Field(default=500, ge=0)
    max_items_per_run: int | None = Field(default=None, ge=0)


class TumblrConfig(BaseModel):
    tags: list[str] = Field(min_length=1)
    api_key: str = Field(min_length=1)
    min_note_count: int = Field(default=100, ge=0)
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

    def parse(self, raw: dict[str, Any], config: dict[str, Any]) -> list[DiscoveredItem]:
        cfg = MemeApiConfig.model_validate(config)
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

            score = meme.get("ups")
            if not isinstance(score, int) or isinstance(score, bool) or score < cfg.min_score:
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


def _tumblr_media_urls(post: dict[str, Any]) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []

    photos = post.get("photos")
    if isinstance(photos, list):
        for photo in photos:
            if not isinstance(photo, dict):
                continue
            alt_sizes = photo.get("alt_sizes")
            if not isinstance(alt_sizes, list):
                continue
            for image in alt_sizes:
                if not isinstance(image, dict):
                    continue
                url = image.get("url")
                if not isinstance(url, str) or not is_still_image_url(url):
                    continue
                width = image.get("width")
                candidates.append((width if isinstance(width, int) else 0, url))

    content = post.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image":
                continue
            media = block.get("media")
            media_items = media if isinstance(media, list) else [media]
            for image in media_items:
                if not isinstance(image, dict):
                    continue
                url = image.get("url")
                if not isinstance(url, str) or not is_still_image_url(url):
                    continue
                width = image.get("width")
                candidates.append((width if isinstance(width, int) else 0, url))

    return candidates


class TumblrTaggedAdapter:
    key = "tumblr_tagged"

    def build_requests(self, config: dict[str, Any], *, rng: Random) -> list[FetchRequest]:
        cfg = TumblrConfig.model_validate(config)

        tags = list(cfg.tags)
        rng.shuffle(tags)
        remaining = (
            cfg.max_items_per_run
            if cfg.max_items_per_run is not None
            else len(tags) * TUMBLR_MAX_PER_CALL
        )

        requests: list[FetchRequest] = []
        for tag in tags:
            if remaining <= 0:
                break
            limit = min(TUMBLR_MAX_PER_CALL, remaining)
            query = urlencode({"tag": tag, "api_key": cfg.api_key, "limit": limit})
            requests.append(
                FetchRequest(
                    url=f"{TUMBLR_TAGGED_URL}?{query}",
                    headers={"user-agent": TUMBLR_USER_AGENT},
                )
            )
            remaining -= limit

        return requests

    def parse(self, raw: dict[str, Any], config: dict[str, Any]) -> list[DiscoveredItem]:
        cfg = TumblrConfig.model_validate(config)
        raw_posts = raw.get("response")
        posts = raw_posts if isinstance(raw_posts, list) else []

        items: list[DiscoveredItem] = []
        for post in posts:
            if not isinstance(post, dict):
                continue

            note_count = post.get("note_count")
            if (
                not isinstance(note_count, int)
                or isinstance(note_count, bool)
                or note_count < cfg.min_note_count
            ):
                continue

            post_id = post.get("id_string")
            if not isinstance(post_id, str) or not post_id:
                post_id_value = post.get("id")
                if not isinstance(post_id_value, int) or isinstance(post_id_value, bool):
                    continue
                post_id = str(post_id_value)

            media_urls = _tumblr_media_urls(post)
            if not media_urls:
                continue
            _, media_url = max(media_urls, key=lambda candidate: candidate[0])

            title = post.get("title")
            if not isinstance(title, str) or not title:
                caption = post.get("caption")
                title = caption if isinstance(caption, str) and caption else None

            items.append(
                DiscoveredItem(
                    external_item_id=post_id,
                    media_url=media_url,
                    canonical_item_url=post.get("post_url"),
                    title=title,
                    raw_metadata={
                        key: post.get(key) for key in _TUMBLR_RAW_METADATA_KEYS if key in post
                    },
                )
            )

        return items


ADAPTERS: dict[str, Adapter] = {
    adapter.key: adapter for adapter in (MemeApiAdapter(), TumblrTaggedAdapter())
}

KNOWN_ADAPTER_KEYS = frozenset(ADAPTERS)


def get_adapter(key: str) -> Adapter:
    try:
        return ADAPTERS[key]
    except KeyError:
        raise UnknownAdapterKey(key) from None
