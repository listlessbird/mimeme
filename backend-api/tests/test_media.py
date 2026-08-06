"""Public media URL behavior."""

from urllib.parse import parse_qs, urlsplit

import pytest

from mimeme.media import Urls


@pytest.mark.parametrize(
    ("base_url", "key", "expected_path"),
    [
        ("https://assets.mimeme.dev", "images/cats/meme.jpg", "/images/cats/meme.jpg"),
        ("https://assets.mimeme.dev/", "/images/my meme.jpg", "/images/my%20meme.jpg"),
        ("https://assets.mimeme.dev", "images/100%/why?.jpg", "/images/100%25/why%3F.jpg"),
        ("https://assets.mimeme.dev", "images/#wow/猫.png", "/images/%23wow/%E7%8C%AB.png"),
    ],
)
def test_media_url_resolver_builds_permanent_encoded_urls(
    base_url: str, key: str, expected_path: str
) -> None:
    url = Urls(base_url).resolve(key)
    parsed = urlsplit(url)

    assert parsed.scheme == "https"
    assert parsed.netloc == "assets.mimeme.dev"
    assert parsed.path == expected_path
    assert parse_qs(parsed.query) == {}
    assert parsed.fragment == ""


def test_media_url_resolver_rejects_an_empty_key() -> None:
    with pytest.raises(ValueError, match="media key"):
        Urls("https://assets.mimeme.dev").resolve("")
