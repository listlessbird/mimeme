import random
import re
from collections.abc import Mapping
from typing import Any

import httpx
import structlog
from botocore.compat import urlparse

from activities.sources.models import FetchSourceItemsOutput, MemeApiResponse, SourceItemData

log = structlog.get_logger()

# matches both https://redd.it/abc123 and https://reddit.com/r/sub/comments/abc123/...
_POST_ID_PATTERN = re.compile(r"(?:redd\.it|/comments/)/([a-z0-9]+)")

_REJECTED_EXTENSIONS = frozenset((".gif", ".gifv", ".mp4", ".webm"))
_IMAGE_EXTENSIONS = frozenset((".jpg", ".jpeg", ".png", ".webp"))


def _extract_reddit_post_id(post_link: str) -> str | None:
    match = _POST_ID_PATTERN.search(post_link)
    return match.group(1) if match else None


def _validate_image_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    filename = path.rsplit("/", 1)[-1]

    if any(path.endswith(ext) for ext in _REJECTED_EXTENSIONS):
        return False

    has_known_ext = any(path.endswith(ext) for ext in _IMAGE_EXTENSIONS)
    has_no_ext = "." not in filename

    return has_known_ext or has_no_ext


class MemeApiAdaper:
    secret_ref_names: tuple[str, ...] = ()

    def fetch_latest(
        self,
        adapter_cfg: dict[str, Any],
        max_items: int,
        secrets: Mapping[str, str],
    ) -> FetchSourceItemsOutput:
        subreddits: list[str] = adapter_cfg.get("subreddits", ["memes"])
        batch_sz: int = adapter_cfg.get("batch_size_per_call", 20)
        allow_nsfw: bool = adapter_cfg.get("allow_nsfw", False)

        media_policy: str = adapter_cfg.get("media_policy", "images")

        shuffled: list[str] = random.sample(subreddits, len(subreddits))

        accepted = []
        skipped = 0
        seen_post_ids: set[str] = set()
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            for subreddit in shuffled:
                if len(accepted) >= max_items:
                    break

                remaining = max_items - len(accepted)
                count = min(batch_sz, remaining + 10)

                url = f"https://meme-api.com/gimme/{subreddit}/{count}"
                try:
                    response = client.get(url)
                    response.raise_for_status()
                    data = MemeApiResponse.model_validate(response.json())
                except (httpx.HTTPError, ValueError) as e:
                    log.warning(
                        "meme_api_request_failed",
                        url=url,
                        subreddit=subreddit,
                        error=str(e),
                        status_code=e.response.status_code
                        if isinstance(e, httpx.HTTPStatusError)
                        else None,
                    )
                    continue

                memes = data.memes

                for meme in memes:
                    if len(accepted) >= max_items:
                        break

                    post_link = meme.post_link
                    post_id = _extract_reddit_post_id(post_link) or post_link.split("/")[-1]

                    if post_id in seen_post_ids:
                        continue
                    seen_post_ids.add(post_id)

                    if not allow_nsfw and meme.nsfw:
                        skipped += 1
                        continue

                    image_url = meme.url

                    if media_policy == "images" and not _validate_image_url(image_url):
                        skipped += 1
                        continue

                    accepted.append(
                        SourceItemData(
                            external_item_id=post_id,
                            canonical_item_url=f"https://www.reddit.com/comments/{post_id}",
                            fetch_url=image_url,
                            canonical_media_id=f"reddit-post:{post_id}",
                            media_type="image",
                            title=meme.title,
                            raw_metadata=meme.model_dump(exclude={"url", "post_link"}),
                        )
                    )

        return FetchSourceItemsOutput(items=accepted, skipped=skipped)
