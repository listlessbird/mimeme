from __future__ import annotations

from random import Random

import pytest
from pydantic import ValidationError

from mimeme.source.adapter import (
    KNOWN_ADAPTER_KEYS,
    MemeApiAdapter,
    UnknownAdapterKey,
    get_adapter,
)


class TestRegistry:
    def test_known_key_resolves(self) -> None:
        assert get_adapter("meme_api").key == "meme_api"
        assert "meme_api" in KNOWN_ADAPTER_KEYS

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(UnknownAdapterKey):
            get_adapter("nope")


class TestBuildRequests:
    def test_bounds_by_max_items(self) -> None:
        requests = MemeApiAdapter().build_requests(
            {"subreddits": ["a", "b", "c"], "max_items_per_run": 60}, rng=Random(0)
        )
        # 60 items over 50-per-call cap -> one full call (50) + one partial (10).
        total = sum(int(r.url.rsplit("/", 1)[-1]) for r in requests)
        assert total == 60

    def test_one_request_per_subreddit_uncapped(self) -> None:
        requests = MemeApiAdapter().build_requests(
            {"subreddits": ["a", "b"]}, rng=Random(0)
        )
        assert len(requests) == 2
        assert all(r.url.endswith("/50") for r in requests)

    def test_rejects_empty_subreddits(self) -> None:
        with pytest.raises(ValidationError):
            MemeApiAdapter().build_requests({"subreddits": []}, rng=Random(0))


class TestParse:
    def test_extracts_still_images_only(self) -> None:
        raw = {
            "memes": [
                {
                    "postLink": "https://reddit.com/r/memes/comments/abc/x",
                    "url": "https://i.redd.it/abc.jpg",
                    "title": "ok",
                    "nsfw": False,
                },
                {  # video extension -> dropped
                    "postLink": "https://reddit.com/r/memes/comments/def/x",
                    "url": "https://v.redd.it/def.mp4",
                },
                {  # nsfw -> dropped
                    "postLink": "https://reddit.com/r/memes/comments/ghi/x",
                    "url": "https://i.redd.it/ghi.jpg",
                    "nsfw": True,
                },
                {  # no post link -> dropped
                    "url": "https://i.redd.it/xyz.jpg",
                },
            ]
        }
        items = MemeApiAdapter().parse(raw)
        assert [i.external_item_id for i in items] == ["abc"]
        assert items[0].media_url == "https://i.redd.it/abc.jpg"

    def test_single_meme_shape(self) -> None:
        raw = {
            "postLink": "https://reddit.com/r/memes/comments/solo/x",
            "url": "https://i.redd.it/solo.png",
        }
        items = MemeApiAdapter().parse(raw)
        assert len(items) == 1 and items[0].external_item_id == "solo"
