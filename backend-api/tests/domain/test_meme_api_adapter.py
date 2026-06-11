"""Executable spec for the pure ``meme_api`` Adapter (issue 03, slice 1).

The Adapter is a pure domain strategy (ADR 0001): no HTTP lives here. It does two
things, both unit-testable without mocking I/O:

    MemeApiAdapter().build_requests(config, *, rng) -> list[FetchRequest]
        Plan the per-run batch: shuffle the configured subreddits with the
        injected ``rng`` (stateless, no persisted rotation cursor), then split
        ``max_items_per_run`` across them at <= 50 items per ``/gimme`` call.

    MemeApiAdapter().parse(raw) -> list[DiscoveredItem]
        Turn one provider response into discovered items: map fields, derive a
        stable ``external_item_id`` from ``postLink``, drop the items v1 must not
        ingest (no identity, NSFW, GIF/video), and keep the rest.

The interesting per-provider logic is parsing, which is why it is pure: feed it
representative provider JSON and assert on the items it yields. ``rng`` is
injected into ``build_requests`` so the shuffle is deterministic in tests.

Field mapping (provider meme -> DiscoveredItem):
    postLink  -> external_item_id (parsed) AND canonical_item_url (verbatim)
    url       -> media_url        (fetch target only; downloaded during the run)
    title     -> title
    (none)    -> canonical_image_url is always None for this provider
    {author, title, ups, subreddit, preview, postLink} -> raw_metadata

Prior art: tests/domain/test_phash_gate.py (helper + decision split, pure).
"""

from __future__ import annotations

from typing import Any

import pytest

from domain.adapters.base import DiscoveredItem, FetchRequest
from domain.adapters.meme_api import (
    MemeApiAdapter,
    is_still_image_url,
    parse_external_item_id,
)


def _meme(**overrides: Any) -> dict[str, Any]:
    """A representative provider meme. Override any field per test."""
    meme: dict[str, Any] = {
        "postLink": "https://redd.it/1u2jcm6",
        "subreddit": "DeadlockTheGame",
        "title": "Put yourself in the shoes of our brave troopers.",
        "url": "https://i.redd.it/9dmiranpoj6h1.png",
        "nsfw": False,
        "spoiler": False,
        "author": "xArcadex1243",
        "ups": 358,
        "preview": [
            "https://preview.redd.it/9dmiranpoj6h1.png?width=108&crop=smart&auto=webp&s=189498e4a79542e0ad844cfd428dd468399f2e17",
            "https://preview.redd.it/9dmiranpoj6h1.png?width=216&crop=smart&auto=webp&s=80d994c2d399458cea092bb80b706e61def9b837",
            "https://preview.redd.it/9dmiranpoj6h1.png?width=320&crop=smart&auto=webp&s=3fe0fd7cc9d86d3257221645ee715134606eb7b1",
            "https://preview.redd.it/9dmiranpoj6h1.png?width=640&crop=smart&auto=webp&s=c5eebde3750b3ddc37f94e26593b4550dd3dcdb2",
            "https://preview.redd.it/9dmiranpoj6h1.png?width=960&crop=smart&auto=webp&s=12e7a3642ad263a7dfd9170ebd7e8f7a32098fe9",
            "https://preview.redd.it/9dmiranpoj6h1.png?width=1080&crop=smart&auto=webp&s=240d2bf82fd3c3539d60402d8898d104f7e127aa",
        ],
    }
    meme.update(overrides)
    return meme


def _response(*memes: dict[str, Any]) -> dict[str, Any]:
    """The ``/gimme/{subreddit}/{count}`` envelope: ``{count, memes: [...]}``."""
    return {"count": len(memes), "memes": list(memes)}


def _fixed_rng(order: list[str]) -> Any:
    """An rng stub whose ``shuffle`` reorders a list into ``order``.

    Lets a test pin the post-shuffle subreddit order without depending on the
    stdlib RNG internals — the Adapter only contracts to call ``rng.shuffle``.
    """

    class _FixedRng:
        def shuffle(self, seq: list[str]) -> None:
            seq.sort(key=order.index)

    return _FixedRng()


def _adapter() -> MemeApiAdapter:
    return MemeApiAdapter()


def test_one_request_per_subreddit_in_shuffled_order() -> None:
    requests = _adapter().build_requests(
        {"subreddits": ["a", "b", "c"], "max_items_per_run": 150},
        rng=_fixed_rng(["c", "a", "b"]),
    )

    assert [r.url for r in requests] == [
        "https://meme-api.com/gimme/c/50",
        "https://meme-api.com/gimme/a/50",
        "https://meme-api.com/gimme/b/50",
    ]


def test_requests_are_get() -> None:
    requests = _adapter().build_requests(
        {"subreddits": ["a"], "max_items_per_run": 10},
        rng=_fixed_rng(["a"]),
    )

    assert all(r.method == "GET" for r in requests)
    assert isinstance(requests[0], FetchRequest)


def test_each_call_is_capped_at_fifty() -> None:
    requests = _adapter().build_requests(
        {"subreddits": ["a"], "max_items_per_run": 50},
        rng=_fixed_rng(["a"]),
    )

    # One subreddit, one call; the provider allows at most 50 per /gimme call.
    assert [r.url for r in requests] == ["https://meme-api.com/gimme/a/50"]


def test_max_items_split_across_subreddits_with_partial_last_call() -> None:
    requests = _adapter().build_requests(
        {"subreddits": ["a", "b", "c"], "max_items_per_run": 120},
        rng=_fixed_rng(["a", "b", "c"]),
    )

    assert [r.url for r in requests] == [
        "https://meme-api.com/gimme/a/50",
        "https://meme-api.com/gimme/b/50",
        "https://meme-api.com/gimme/c/20",
    ]


def test_stops_once_max_items_reached_leaving_extra_subreddits_unused() -> None:
    requests = _adapter().build_requests(
        {"subreddits": ["a", "b", "c", "d"], "max_items_per_run": 30},
        rng=_fixed_rng(["a", "b", "c", "d"]),
    )

    # 30 items fit in the first call; b/c/d are not fetched this run.
    assert [r.url for r in requests] == ["https://meme-api.com/gimme/a/30"]


def test_fewer_subreddits_than_max_yields_one_call_each() -> None:
    requests = _adapter().build_requests(
        {"subreddits": ["a"], "max_items_per_run": 120},
        rng=_fixed_rng(["a"]),
    )

    # A run spreads across subreddits (one call each), never re-calls one to
    # backfill the cap — so a single subreddit yields a single capped call.
    assert [r.url for r in requests] == ["https://meme-api.com/gimme/a/50"]


def test_missing_max_items_defaults_to_one_capped_call_per_subreddit() -> None:
    requests = _adapter().build_requests(
        {"subreddits": ["a", "b"], "max_items_per_run": None},
        rng=_fixed_rng(["a", "b"]),
    )

    assert [r.url for r in requests] == [
        "https://meme-api.com/gimme/a/50",
        "https://meme-api.com/gimme/b/50",
    ]


def test_empty_subreddits_is_a_config_error() -> None:
    with pytest.raises(ValueError):
        _adapter().build_requests(
            {"subreddits": [], "max_items_per_run": 50},
            rng=_fixed_rng([]),
        )


def test_parse_maps_all_fields() -> None:
    meme = _meme()
    [item] = _adapter().parse(_response(meme))

    assert isinstance(item, DiscoveredItem)
    assert item.external_item_id == "1u2jcm6"
    assert item.media_url == meme["url"]
    assert item.canonical_item_url == meme["postLink"]
    assert item.canonical_image_url is None
    assert item.title == meme["title"]


def test_parse_raw_metadata_retains_exactly_the_specified_keys() -> None:
    meme = _meme()
    [item] = _adapter().parse(_response(meme))

    assert set(item.raw_metadata) == {
        "author",
        "title",
        "ups",
        "subreddit",
        "preview",
        "postLink",
    }
    assert item.raw_metadata["author"] == meme["author"]
    assert item.raw_metadata["ups"] == meme["ups"]
    assert item.raw_metadata["subreddit"] == meme["subreddit"]
    assert item.raw_metadata["postLink"] == meme["postLink"]


def test_parse_empty_response_yields_no_items() -> None:
    assert _adapter().parse(_response()) == []


def test_external_item_id_from_redd_it_short_link() -> None:
    [item] = _adapter().parse(_response(_meme(postLink="https://redd.it/lr3g8z")))

    assert item.external_item_id == "lr3g8z"


def test_external_item_id_from_full_reddit_comments_link() -> None:
    # The provider also returns the long form; the post id is the comments id.
    link = "https://www.reddit.com/r/ProgrammerHumor/comments/lr3g8z/some_slug/"
    [item] = _adapter().parse(_response(_meme(postLink=link)))

    assert item.external_item_id == "lr3g8z"


def test_item_with_unparseable_post_link_is_dropped() -> None:
    # No URL fallback for identity — an item without a stable id is dropped.
    assert _adapter().parse(_response(_meme(postLink="https://example.com/foo"))) == []


def test_item_with_missing_post_link_is_dropped() -> None:
    assert _adapter().parse(_response(_meme(postLink=""))) == []


def test_nsfw_item_is_dropped() -> None:
    assert _adapter().parse(_response(_meme(nsfw=True))) == []


def test_spoiler_item_is_kept() -> None:
    [item] = _adapter().parse(_response(_meme(spoiler=True)))

    assert item.external_item_id == "1u2jcm6"


def test_gif_media_is_dropped() -> None:
    assert _adapter().parse(_response(_meme(url="https://i.redd.it/abc123.gif"))) == []


def test_video_media_is_dropped() -> None:
    assert _adapter().parse(_response(_meme(url="https://i.redd.it/abc123.mp4"))) == []


def test_still_image_is_kept() -> None:
    for url in (
        "https://i.redd.it/abc123.jpg",
        "https://i.redd.it/abc123.jpeg",
        "https://i.redd.it/abc123.png",
        "https://i.redd.it/abc123.webp",
    ):
        [item] = _adapter().parse(_response(_meme(url=url)))
        assert item.media_url == url


def test_media_extension_classified_ignoring_query_string() -> None:
    gif = "https://i.redd.it/abc123.gif?width=640&format=mp4"
    assert _adapter().parse(_response(_meme(url=gif))) == []

    png = "https://i.redd.it/abc123.png?width=640&auto=webp"
    [item] = _adapter().parse(_response(_meme(url=png)))
    assert item.media_url == png


def test_mixed_batch_keeps_only_valid_items_in_order() -> None:
    items = _adapter().parse(
        _response(
            _meme(postLink="https://redd.it/keep1"),
            _meme(nsfw=True, postLink="https://redd.it/nsfw"),
            _meme(url="https://i.redd.it/x.gif", postLink="https://redd.it/gif"),
            _meme(postLink="https://example.com/no-id"),
            _meme(spoiler=True, postLink="https://redd.it/keep2"),
        )
    )

    assert [i.external_item_id for i in items] == ["keep1", "keep2"]


def test_parse_external_item_id_handles_both_link_forms() -> None:
    assert parse_external_item_id("https://redd.it/lr3g8z") == "lr3g8z"
    assert (
        parse_external_item_id("https://www.reddit.com/r/memes/comments/lr3g8z/slug/") == "lr3g8z"
    )


def test_parse_external_item_id_returns_none_when_no_id() -> None:
    assert parse_external_item_id("") is None
    assert parse_external_item_id("https://example.com/foo") is None


def test_is_still_image_url_blocks_gif_and_video() -> None:
    assert is_still_image_url("https://i.redd.it/x.png") is True
    assert is_still_image_url("https://i.redd.it/x.gif") is False
    assert is_still_image_url("https://i.redd.it/x.mp4") is False
    assert is_still_image_url("https://v.redd.it/x") is False
