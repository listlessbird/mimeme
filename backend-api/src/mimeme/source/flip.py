from __future__ import annotations

import re
from collections.abc import AsyncIterator
from enum import StrEnum
from random import Random
from urllib.parse import urljoin, urlparse

from pydantic import BaseModel, ConfigDict, Field
from scrapling.parser import Selector

from mimeme.source.adapter import Fetcher
from mimeme.source.model import DiscoveredItem, DiscoveredMedia, KnownFacts, Retryable

BASE_URL = "https://imgflip.com"


class ListingMode(StrEnum):
    TOP_30_DAYS = "top-30-days"
    TOP_ALL_TIME = "top-all-time"
    TOP_NEW = "top-new"


LISTING_URLS: dict[ListingMode, str] = {
    ListingMode.TOP_30_DAYS: f"{BASE_URL}/memetemplates",
    ListingMode.TOP_ALL_TIME: f"{BASE_URL}/memetemplates?sort=top-all-time",
    ListingMode.TOP_NEW: f"{BASE_URL}/memetemplates?sort=top-new",
}

_IMAGE_SUFFIXES = frozenset({".avif", ".jpeg", ".jpg", ".png", ".webp"})


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FlipConfig(_Frozen):
    mode: ListingMode = ListingMode.TOP_30_DAYS
    max_templates_per_run: int = Field(default=15, ge=1, le=100)
    max_meme_pages: int = Field(default=1, ge=1, le=20)
    max_items_per_run: int | None = Field(default=None, ge=0)
    delay_seconds: float = Field(default=1.0, ge=0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    retries: int = Field(default=3, ge=0, le=10)
    impersonate: str = "chrome"


class TemplateRef(_Frozen):
    title: str
    url: str
    thumbnail_url: str


class MediaCandidate(_Frozen):
    external_media_id: str
    media_url: str
    raw_metadata: dict[str, object]


class FlipAdapter:
    key = "flip"

    async def discover(
        self, config: dict[str, object], *, fetcher: Fetcher, rng: Random
    ) -> AsyncIterator[DiscoveredItem]:
        del rng
        cfg = FlipConfig.model_validate(config)
        remaining = cfg.max_items_per_run
        if remaining == 0:
            return

        refs = await _listing(fetcher, cfg)
        for ref in refs:
            item = await _template(fetcher, ref, cfg)
            if item is None:
                continue
            if remaining is not None:
                item = item.model_copy(update={"media": item.media[:remaining]})
                remaining -= len(item.media)
            if item.media:
                yield item
            if remaining == 0:
                break


async def _listing(fetcher: Fetcher, cfg: FlipConfig) -> list[TemplateRef]:
    refs: list[TemplateRef] = []
    seen_urls: set[str] = set()
    page_url: str | None = LISTING_URLS[cfg.mode]

    while page_url is not None and len(refs) < cfg.max_templates_per_run:
        page = Selector(await fetcher.html(page_url), url=page_url)
        cards = page.css(".mt-box")
        if not cards and not refs:
            raise Retryable(f"Imgflip listing selector matched no templates at {page_url}")

        for card in cards:
            title_link = card.css("h3.mt-title a[href]").first
            image = card.css(".mt-img-wrap img").first
            if title_link is None or image is None:
                continue
            href = _attribute(title_link, "href")
            thumbnail = _attribute(image, "src")
            if href is None or thumbnail is None or not href.startswith("/meme/"):
                continue
            template_url = urljoin(BASE_URL, href)
            thumbnail_url = urljoin(BASE_URL, thumbnail)
            if template_url in seen_urls or not _is_image_url(thumbnail_url):
                continue
            seen_urls.add(template_url)
            refs.append(
                TemplateRef(
                    title=_clean(title_link.get_all_text(strip=True)),
                    url=template_url,
                    thumbnail_url=thumbnail_url,
                )
            )
            if len(refs) >= cfg.max_templates_per_run:
                break

        next_href = page.css("a.pager-next[href]::attr(href)").get()
        page_url = urljoin(BASE_URL, next_href) if isinstance(next_href, str) else None

    return refs


async def _template(fetcher: Fetcher, ref: TemplateRef, cfg: FlipConfig) -> DiscoveredItem | None:
    hot = Selector(await fetcher.html(ref.url), url=ref.url)
    aliases = _aliases(hot)
    candidates: dict[str, MediaCandidate] = {}

    blank = _blank(hot)
    if blank is not None:
        candidates[blank.media_url] = blank

    await _add_feed(
        fetcher,
        candidates,
        feed="hot",
        start_url=ref.url,
        first_page=hot,
        max_pages=cfg.max_meme_pages,
    )

    latest_href = hot.css(".base-options .base-latest[href]::attr(href)").get()
    if isinstance(latest_href, str):
        await _add_feed(
            fetcher,
            candidates,
            feed="new",
            start_url=urljoin(BASE_URL, latest_href),
            first_page=None,
            max_pages=cfg.max_meme_pages,
        )

    if not candidates:
        return None

    media = [
        DiscoveredMedia(
            external_media_id=candidate.external_media_id,
            media_url=candidate.media_url,
            canonical_media_url=candidate.media_url,
            raw_metadata=candidate.raw_metadata,
        )
        for candidate in candidates.values()
    ]
    return DiscoveredItem(
        external_item_id=ref.url,
        canonical_item_url=ref.url,
        title=ref.title,
        known_facts=KnownFacts(
            title=ref.title,
            tags=aliases,
            types=["meme template family"],
            origin="Imgflip",
        ),
        raw_metadata={
            "aliases": aliases,
            "listing_mode": cfg.mode.value,
            "thumbnail_url": ref.thumbnail_url,
        },
        media=media,
    )


async def _add_feed(
    fetcher: Fetcher,
    candidates: dict[str, MediaCandidate],
    *,
    feed: str,
    start_url: str,
    first_page: Selector | None,
    max_pages: int,
) -> None:
    page_url: str | None = start_url
    for _page_number in range(1, max_pages + 1):
        if page_url is None:
            break
        page = first_page or Selector(await fetcher.html(page_url), url=page_url)
        first_page = None

        for unit in page.css(".base-unit"):
            candidate = _example(unit, feed=feed)
            if candidate is None:
                continue
            previous = candidates.get(candidate.media_url)
            if previous is None:
                candidates[candidate.media_url] = candidate
                continue
            feeds = previous.raw_metadata.get("feeds")
            previous_feeds = feeds if isinstance(feeds, list) else []
            candidates[candidate.media_url] = previous.model_copy(
                update={
                    "raw_metadata": {
                        **previous.raw_metadata,
                        "feeds": list(dict.fromkeys([*previous_feeds, feed])),
                    }
                }
            )

        next_href = page.css("a.pager-next[href]::attr(href)").get()
        page_url = urljoin(BASE_URL, next_href) if isinstance(next_href, str) else None


def _blank(page: Selector) -> MediaCandidate | None:
    image = page.css("a.meme-link[href^='/memetemplate/'] img").first
    if image is None:
        return None
    value = _attribute(image, "src") or _attribute(image, "data-src")
    if value is None:
        return None
    media_url = urljoin(BASE_URL, value)
    if not _is_image_url(media_url):
        return None
    return MediaCandidate(
        external_media_id="blank",
        media_url=media_url,
        raw_metadata={"alt": _clean(_attribute(image, "alt")) or None, "role": "blank_template"},
    )


def _example(unit: Selector, *, feed: str) -> MediaCandidate | None:
    title_link = unit.css("h2.base-unit-title a[href]").first
    media = unit.css(".base-img").first
    if title_link is None or media is None:
        return None
    href = _attribute(title_link, "href")
    value = _attribute(media, "src") or _attribute(media, "data-src")
    if href is None or value is None or not href.startswith("/i/"):
        return None
    media_url = urljoin(BASE_URL, value)
    if not _is_image_url(media_url):
        return None

    stats = _text(unit.css(".base-view-count").first)
    alt = _attribute(media, "alt") or _attribute(media, "data-alt")
    return MediaCandidate(
        external_media_id=href.removeprefix("/i/").strip("/"),
        media_url=media_url,
        raw_metadata={
            "alt": _clean(alt) or None,
            "author": _text(unit.css(".base-author a.u-username").first),
            "comments": _metric(stats, "comment"),
            "feeds": [feed],
            "page_url": urljoin(BASE_URL, href),
            "role": "example_meme",
            "stats": stats,
            "tags": _tags(alt),
            "title": _clean(title_link.get_all_text(strip=True)),
            "upvotes": _metric(stats, "upvote"),
            "views": _metric(stats, "view"),
        },
    )


def _aliases(page: Selector) -> list[str]:
    element = page.css(".alt-names").first
    if element is None:
        return []
    value = re.sub(r"^aka:\s*", "", _clean(element.get_all_text(strip=True)), flags=re.IGNORECASE)
    return [alias.strip() for alias in value.split(",") if alias.strip()]


def _tags(alt: str | None) -> list[str]:
    if not alt:
        return []
    match = re.search(r"\| image tagged in (.*?) \| made w/", alt, flags=re.IGNORECASE)
    return [tag.strip() for tag in match.group(1).split(",") if tag.strip()] if match else []


def _metric(stats: str | None, name: str) -> int | None:
    if stats is None:
        return None
    match = re.search(rf"([\d,]+) {name}s?\b", stats)
    return int(match.group(1).replace(",", "")) if match else None


def _is_image_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and any(
        parsed.path.lower().endswith(suffix) for suffix in _IMAGE_SUFFIXES
    )


def _attribute(element: Selector, name: str) -> str | None:
    value = element.attrib.get(name)
    return value if isinstance(value, str) and value.strip() else None


def _text(element: Selector | None) -> str | None:
    if element is None:
        return None
    value = _clean(element.get_all_text(strip=True))
    return value or None


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())
