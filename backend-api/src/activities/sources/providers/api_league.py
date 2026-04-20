from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog

from activities.sources.models import ApiLeagueMeme, FetchSourceItemsOutput, SourceItemData

log = structlog.get_logger()

_API_LEAGUE_URL = "https://api.apileague.com/retrieve-random-meme"


def _normalize_media_id(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    host = parsed.hostname or ""
    path = parsed.path.rstrip("/") or "/"
    return f"{scheme}://{host}{path}"


def _build_external_item_id(canonical_media_id: str) -> str:
    digest = hashlib.sha256(canonical_media_id.encode()).hexdigest()
    return f"api_league:{digest}"


class ApiLeagueAdapter:
    secret_ref_names = ("api_league_api_key",)

    def fetch_latest(
        self, adapter_cfg: dict[str, Any], max_items: int, secrets: Mapping[str, str]
    ) -> FetchSourceItemsOutput:
        api_key = secrets["api_league_api_key"]

        attempt_multiplier: int = adapter_cfg.get("attempt_multiplier", 3)
        max_fetch_attempts = attempt_multiplier * max_items
        min_interval_ms: int = adapter_cfg.get("min_request_interval_ms", 1100)
        media_policy: str = adapter_cfg.get("media_policy", "images")

        params: dict[str, Any] = {"api-key": api_key}
        if "max-age-days" in adapter_cfg:
            params["max-age-days"] = adapter_cfg["max-age-days"]
        if "min-rating" in adapter_cfg:
            params["min-rating"] = adapter_cfg["min-rating"]
        if "media-type" in adapter_cfg:
            params["media-type"] = adapter_cfg["media-type"]

        accepted: list[SourceItemData] = []
        skipped = 0
        seen_media_ids: set[str] = set()
        provider_state: dict[str, Any] = {}
        attempts = 0

        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            while len(accepted) < max_items and attempts < max_fetch_attempts:
                attempts += 1

                try:
                    response = client.get(_API_LEAGUE_URL, params=params)
                except httpx.HTTPError as e:
                    log.warning(
                        "api_league_request_error",
                        attempt=attempts,
                        error=str(e),
                        status_code=e.response.status_code
                        if isinstance(e, httpx.HTTPStatusError)
                        else None,
                    )
                    time.sleep(min_interval_ms / 1000)
                    continue

                _capture_quota_headers(response, provider_state)

                if response.status_code == 402:
                    log.warning("api_league_quota_exhausted", attempt=attempts)
                    break

                if response.status_code == 429:
                    backoff = min(2 ** (attempts % 5) + random.uniform(0, 1), 30)
                    log.info("api_league_rate_limited", backoff_s=backoff)
                    time.sleep(backoff)
                    continue

                if response.status_code != 200:
                    log.warning(
                        "api_league_unexpected_status",
                        status=response.status_code,
                        attempt=attempts,
                    )
                    time.sleep(min_interval_ms / 1000)
                    continue

                try:
                    data = ApiLeagueMeme.model_validate(response.json())
                except (ValueError, httpx.DecodingError) as e:
                    log.warning("api_league_invalid_response", attempt=attempts, error=str(e))
                    skipped += 1
                    time.sleep(min_interval_ms / 1000)
                    continue

                if media_policy == "images" and data.type.lower() not in (
                    "image/jpeg",
                    "image/png",
                    "image/webp",
                ):
                    skipped += 1
                    time.sleep(min_interval_ms / 1000)
                    continue

                canonical_media_id = _normalize_media_id(data.url)
                if canonical_media_id in seen_media_ids:
                    time.sleep(min_interval_ms / 1000)
                    continue
                seen_media_ids.add(canonical_media_id)

                external_item_id = _build_external_item_id(canonical_media_id)

                accepted.append(
                    SourceItemData(
                        external_item_id=external_item_id,
                        canonical_item_url=None,
                        fetch_url=data.url,
                        canonical_media_id=canonical_media_id,
                        media_type=data.type,
                        title=data.description,
                        raw_metadata=data.model_dump(),
                    )
                )

                time.sleep(min_interval_ms / 1000)

        status_hint = "ok"
        if attempts >= max_fetch_attempts and len(accepted) < max_items:
            status_hint = "exhausted_attempts"
        if provider_state.get("quota_left") == "0":
            status_hint = "quota_exhausted"

        for item in accepted:
            if item.raw_metadata is None:
                item.raw_metadata = {}
            item.raw_metadata["_provider_state"] = provider_state
            item.raw_metadata["_status_hint"] = status_hint
            break

        return FetchSourceItemsOutput(items=accepted, skipped=skipped)


def _capture_quota_headers(response: httpx.Response, state: dict[str, Any]) -> None:
    for header, key in [
        ("X-API-Quota-Request", "quota_request"),
        ("X-API-Quota-Used", "quota_used"),
        ("X-API-Quota-Left", "quota_left"),
    ]:
        value = response.headers.get(header)
        if value is not None:
            state[key] = value
