from mimeme.api.deps import get_artifact_storage, get_media_storage, get_media_url_resolver
from mimeme.shared.services.api_storage import AsyncApiStorage


def test_storage_dependencies_are_role_specific() -> None:
    media = get_media_storage()
    artifacts = get_artifact_storage()

    assert isinstance(media, AsyncApiStorage)
    assert isinstance(artifacts, AsyncApiStorage)
    assert media.bucket != artifacts.bucket


def test_media_url_dependency_is_permanent_and_unsigned() -> None:
    url = get_media_url_resolver().resolve("images/source/my meme.jpg")

    assert url.endswith("/images/source/my%20meme.jpg")
    assert "?" not in url
