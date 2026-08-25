from __future__ import annotations

from random import Random
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import ValidationError

from mimeme.source.adapter import (
    DEFAULT_TUMBLR_TAGS,
    KNOWN_ADAPTER_KEYS,
    TUMBLR_USER_AGENT,
    MemeApiAdapter,
    TumblrTaggedAdapter,
    UnknownAdapterKey,
    get_adapter,
)


class TestRegistry:
    def test_known_key_resolves(self) -> None:
        assert get_adapter("meme_api").key == "meme_api"
        assert get_adapter("tumblr_tagged").key == "tumblr_tagged"
        assert get_adapter("flip").key == "flip"
        assert "meme_api" in KNOWN_ADAPTER_KEYS
        assert "tumblr_tagged" in KNOWN_ADAPTER_KEYS
        assert "flip" in KNOWN_ADAPTER_KEYS

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
        requests = MemeApiAdapter().build_requests({"subreddits": ["a", "b"]}, rng=Random(0))
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
                    "ups": 500,
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
        items = MemeApiAdapter().parse(raw, {"subreddits": ["memes"], "min_score": 0})
        assert [i.external_item_id for i in items] == ["abc"]
        assert items[0].media[0].media_url == "https://i.redd.it/abc.jpg"

    def test_single_meme_shape(self) -> None:
        raw = {
            "postLink": "https://reddit.com/r/memes/comments/solo/x",
            "url": "https://i.redd.it/solo.png",
            "ups": 500,
        }
        items = MemeApiAdapter().parse(raw, {"subreddits": ["memes"], "min_score": 0})
        assert len(items) == 1 and items[0].external_item_id == "solo"

    def test_filters_by_min_score(self) -> None:
        raw = {
            "memes": [
                {
                    "postLink": "https://reddit.com/r/memes/comments/low/x",
                    "url": "https://i.redd.it/low.jpg",
                    "ups": 499,
                },
                {
                    "postLink": "https://reddit.com/r/memes/comments/high/x",
                    "url": "https://i.redd.it/high.jpg",
                    "ups": 500,
                },
            ]
        }

        items = MemeApiAdapter().parse(raw, {"subreddits": ["memes"], "min_score": 500})

        assert [item.external_item_id for item in items] == ["high"]

    def test_defaults_min_score_to_500(self) -> None:
        raw = {
            "postLink": "https://reddit.com/r/memes/comments/default/x",
            "url": "https://i.redd.it/default.jpg",
            "ups": 499,
        }

        items = MemeApiAdapter().parse(raw, {"subreddits": ["memes"]})

        assert items == []


class TestTumblrBuildRequests:
    def test_uses_curated_default_tags(self) -> None:
        requests = TumblrTaggedAdapter().build_requests({"api_key": "tumblr-key"}, rng=Random(0))

        tags = {parse_qs(urlsplit(request.url).query)["tag"][0] for request in requests}

        assert tags == set(DEFAULT_TUMBLR_TAGS)

    def test_caps_each_tag_request_at_20_and_sets_user_agent(self) -> None:
        requests = TumblrTaggedAdapter().build_requests(
            {
                "tags": ["meme", "reaction image"],
                "api_key": "tumblr-key",
                "max_items_per_run": 25,
            },
            rng=Random(0),
        )

        assert len(requests) == 2
        assert sorted(
            int(parse_qs(urlsplit(request.url).query)["limit"][0]) for request in requests
        ) == [5, 20]
        for request in requests:
            params = parse_qs(urlsplit(request.url).query)
            assert params["api_key"] == ["tumblr-key"]
            assert request.headers["user-agent"] == TUMBLR_USER_AGENT

    def test_requires_tag_and_api_key(self) -> None:
        with pytest.raises(ValidationError):
            TumblrTaggedAdapter().build_requests({"tags": [], "api_key": "key"}, rng=Random(0))
        with pytest.raises(ValidationError):
            TumblrTaggedAdapter().build_requests({"tags": ["meme"]}, rng=Random(0))


class TestTumblrParse:
    def test_filters_notes_and_selects_largest_still_photo(self) -> None:
        raw = {
            "response": [
                {
                    "id_string": "1234567890123456",
                    "post_url": "https://example.tumblr.com/post/1234567890123456/meme",
                    "type": "photo",
                    "note_count": 99,
                    "photos": [
                        {
                            "alt_sizes": [
                                {"width": 500, "url": "https://64.media.tumblr.com/s500.jpg"}
                            ]
                        }
                    ],
                },
                {
                    "id_string": "1234567890123457",
                    "post_url": "https://example.tumblr.com/post/1234567890123457/meme",
                    "type": "photo",
                    "note_count": 100,
                    "caption": "a reaction image",
                    "tags": ["meme", "reaction image"],
                    "photos": [
                        {
                            "alt_sizes": [
                                {"width": 500, "url": "https://64.media.tumblr.com/s500.jpg"},
                                {"width": 1280, "url": "https://64.media.tumblr.com/s1280.jpg"},
                            ]
                        }
                    ],
                },
                {
                    "id_string": "1234567890123458",
                    "note_count": 500,
                    "type": "text",
                    "photos": [],
                },
            ]
        }

        items = TumblrTaggedAdapter().parse(
            raw, {"tags": ["meme"], "api_key": "key", "min_note_count": 100}
        )

        assert len(items) == 1
        assert items[0].external_item_id == "1234567890123457"
        assert items[0].media[0].media_url.endswith("s1280.jpg")
        assert items[0].canonical_item_url == raw["response"][1]["post_url"]
        assert items[0].title == "a reaction image"
        assert items[0].raw_metadata["note_count"] == 100

    def test_supports_npf_image_blocks_and_numeric_id_fallback(self) -> None:
        raw = {
            "response": [
                {
                    "id": 42,
                    "note_count": 250,
                    "content": [
                        {
                            "type": "image",
                            "media": [
                                {"width": 900, "url": "https://64.media.tumblr.com/image.png"}
                            ],
                        }
                    ],
                }
            ]
        }

        items = TumblrTaggedAdapter().parse(raw, {"tags": ["meme"], "api_key": "key"})

        assert [(item.external_item_id, item.media[0].media_url) for item in items] == [
            ("42", "https://64.media.tumblr.com/image.png")
        ]

    def test_rejects_missing_or_boolean_note_count(self) -> None:
        raw = {
            "response": [
                {"id_string": "missing", "note_count": None},
                {"id_string": "boolean", "note_count": True},
            ]
        }

        items = TumblrTaggedAdapter().parse(raw, {"tags": ["meme"], "api_key": "key"})

        assert items == []
