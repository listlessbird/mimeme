from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlparse

from pydantic import BaseModel, Field
from scrapling.parser import Selector

from mimeme.source.fetch import PageNotFound
from mimeme.source.model import DiscoveredItem, DiscoveredMedia, KnownFacts

if TYPE_CHECKING:
    from mimeme.source.adapter import Fetcher

BASE_URL = "https://knowyourmeme.com"
LISTING_URL = f"{BASE_URL}/memes?kind=confirmed&sort=newest"


class KymConfig(BaseModel):
    start_page: int = Field(default=1, ge=1)
    max_entries_per_run: int = Field(default=5, ge=1)
    max_photo_pages_per_entry: int = Field(default=100, ge=1, le=500)
    max_items_per_run: int | None = Field(default=None, ge=0)
    delay_seconds: float = Field(default=1.0, ge=0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    retries: int = Field(default=3, ge=0, le=10)
    impersonate: str = "chrome"


@dataclass(frozen=True)
class EntryRef:
    slug: str
    title: str
    url: str


class KymAdapter:
    key = "kym"

    async def discover(
        self, config: dict[str, Any], *, fetcher: Fetcher, rng: Random
    ) -> AsyncIterator[DiscoveredItem]:
        del rng
        cfg = KymConfig.model_validate(config)
        entries = await self._listing(fetcher, cfg)
        remaining = cfg.max_items_per_run
        for entry in entries:
            if remaining is not None and remaining <= 0:
                break
            item = await self._entry(fetcher, entry, cfg, remaining=remaining)
            if item is None:
                continue
            yield item
            if remaining is not None:
                remaining -= len(item.media)

    async def _listing(self, fetcher: Fetcher, cfg: KymConfig) -> list[EntryRef]:
        entries: list[EntryRef] = []
        seen: set[str] = set()
        page_number = cfg.start_page
        while len(entries) < cfg.max_entries_per_run:
            url = (
                LISTING_URL
                if page_number == 1
                else f"{BASE_URL}/memes/page/{page_number}?kind=confirmed&sort=newest"
            )
            page = Selector(await fetcher.html(url), url=url)
            found = 0
            for anchor in page.css("section.gallery .groups > a.item[href]"):
                href = anchor.attrib.get("href")
                if not isinstance(href, str) or not href.startswith("/memes/"):
                    continue
                slug = href.removeprefix("/memes/").strip("/")
                if (
                    not slug
                    or slug in seen
                    or slug
                    in {
                        "new",
                        "submissions",
                        "confirmed",
                        "newsworthy",
                        "deadpool",
                    }
                ):
                    continue
                seen.add(slug)
                found += 1
                title = anchor.attrib.get("data-title") or anchor.get_all_text(strip=True)
                entries.append(
                    EntryRef(slug=slug, title=_clean(title), url=urljoin(BASE_URL, href))
                )
                if len(entries) >= cfg.max_entries_per_run:
                    break
            if found == 0:
                break
            page_number += 1
        return entries

    async def _entry(
        self,
        fetcher: Fetcher,
        entry: EntryRef,
        cfg: KymConfig,
        *,
        remaining: int | None,
    ) -> DiscoveredItem | None:
        page = Selector(await fetcher.html(entry.url), url=entry.url)
        sidebar = _sidebar(page)
        media: list[DiscoveredMedia] = []
        seen_media: set[str] = set()
        for page_number in range(1, cfg.max_photo_pages_per_entry + 1):
            if remaining is not None and len(media) >= remaining:
                break
            page_url = (
                f"{entry.url}/photos"
                if page_number == 1
                else f"{entry.url}/photos/page/{page_number}"
            )
            try:
                photos = Selector(await fetcher.html(page_url), url=page_url)
            except PageNotFound:
                break
            gallery = photos.css("article.gallery").first
            if gallery is None:
                break
            page_added = 0
            for image in gallery.css("img"):
                media_url = _image_url(image)
                if media_url is None:
                    continue
                external_media_id = _media_id(media_url)
                if external_media_id in seen_media:
                    continue
                seen_media.add(external_media_id)
                media.append(
                    DiscoveredMedia(
                        external_media_id=external_media_id,
                        media_url=media_url,
                        canonical_media_url=media_url,
                        raw_metadata={
                            "page_url": page_url,
                            "page": page_number,
                            "alt": _clean(image.attrib.get("alt")),
                            "author": _clean(image.attrib.get("data-author")),
                        },
                    )
                )
                page_added += 1
                if remaining is not None and len(media) >= remaining:
                    break
            if gallery.attrib.get("data-end") == "true" or page_added == 0:
                break
        if not media:
            return None
        title = _entry_title(page, entry.title)
        facts = KnownFacts(
            title=title,
            description=_about(page),
            tags=sidebar["tags"],
            categories=sidebar["categories"],
            types=sidebar["types"],
            origin=sidebar["origin"],
            year=sidebar["year"],
        )
        return DiscoveredItem(
            external_item_id=entry.slug,
            canonical_item_url=entry.url,
            title=title,
            known_facts=facts,
            raw_metadata={"status": sidebar["status"], "regions": sidebar["regions"]},
            media=media,
        )


def _about(page: Selector) -> str | None:
    heading = page.css("h2#about").first
    if heading is None:
        return None
    parts: list[str] = []
    for sibling in heading.siblings:
        if sibling.tag == "h2":
            break
        if sibling.tag in {"p", "blockquote", "ul", "ol"}:
            value = _clean(sibling.get_all_text(strip=True))
            if value:
                parts.append(value)
    return "\n\n".join(parts) or None


def _sidebar(page: Selector) -> dict[str, Any]:
    result: dict[str, Any] = {
        "categories": [],
        "status": None,
        "types": [],
        "year": None,
        "origin": None,
        "regions": [],
        "tags": [],
    }
    aside = page.css("aside.left.desktop-only").first
    if aside is None:
        return result
    result["categories"] = [
        _clean(anchor.get_all_text(strip=True)) for anchor in aside.css("a.entry-category-badge")
    ]
    result["tags"] = [
        _clean(anchor.attrib.get("data-tag") or anchor.get_all_text(strip=True))
        for anchor in aside.css("#entry_tags a")
    ]
    key_map = {"type": "types", "region": "regions"}
    for definition_list in aside.css("dl"):
        for term in definition_list.css(":scope > dt"):
            label = _clean(term.get_all_text(strip=True)).rstrip(":").lower()
            definition = term.next
            if definition is None or definition.tag != "dd":
                continue
            values = [_clean(anchor.get_all_text(strip=True)) for anchor in definition.css("a")]
            if not values:
                value = _clean(definition.get_all_text(strip=True))
                values = [value] if value else []
            key = key_map.get(label, label)
            if key in {"types", "regions"}:
                result[key] = values
            elif key in {"status", "year", "origin"}:
                result[key] = values[0] if values else None
    return result


def _image_url(image: Selector) -> str | None:
    value = image.attrib.get("data-image") or image.attrib.get("src")
    if not isinstance(value, str) or not value.startswith("http"):
        encoded = image.attrib.get("data-nsfw-src")
        if isinstance(encoded, str):
            try:
                value = base64.b64decode(encoded).decode()
            except (ValueError, UnicodeDecodeError):
                return None
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    if parsed.netloc != "i.kym-cdn.com" or "/photos/images/original/" not in parsed.path:
        return None
    return value


def _media_id(url: str) -> str:
    marker = "/photos/images/original/"
    return urlparse(url).path.split(marker, 1)[1].rsplit(".", 1)[0]


def _entry_title(page: Selector, fallback: str) -> str:
    for heading in page.css("h1"):
        value = _clean(heading.get_all_text(strip=True))
        if value and value.lower() != "know your meme":
            return value
    return fallback


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())
