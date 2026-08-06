from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from fastapi import Request

from mimeme.api.deps import get_artifact_storage, get_media_storage, get_media_url_resolver
from mimeme.media import Urls
from tests.support.storage import Memory


def _request_with_env(env: object) -> Request:
    return cast(Request, SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(env=env))))


def test_storage_dependencies_are_role_specific() -> None:
    media = Memory()
    artifacts = Memory()
    env = SimpleNamespace(
        media=media,
        artifacts=artifacts,
        media_urls=Urls("https://assets.mimeme.dev"),
    )
    request = _request_with_env(env)

    assert get_media_storage(request) is media
    assert get_artifact_storage(request) is artifacts
    assert get_media_storage(request) is not get_artifact_storage(request)


def test_media_url_dependency_is_permanent_and_unsigned() -> None:
    env = SimpleNamespace(media_urls=Urls("https://assets.mimeme.dev"))
    request = _request_with_env(env)

    url = get_media_url_resolver(request).resolve("images/source/my meme.jpg")

    assert url.endswith("/images/source/my%20meme.jpg")
    assert "?" not in url
