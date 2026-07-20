from __future__ import annotations

from typing import BinaryIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from tests.factories import (
    create_annotation,
    create_artifact,
    create_image,
    create_processing,
    create_search_index_state,
)

from domain.image_catalog import ImageCatalog, ImageCatalogNotFoundError
from shared.models.orm import (
    Annotation,
    Artifact,
    Image,
    Processing,
    ProcessingStatus,
    SearchIndexState,
)
from shared.services.media_url import MediaUrlResolver

pytestmark = pytest.mark.usefixtures(
    "_patch_domain_session_scope", "_patch_async_domain_session_scope"
)

MEDIA_URLS = MediaUrlResolver("https://assets.mimeme.dev")


def catalog(media_storage, artifact_storage=None) -> ImageCatalog:
    return ImageCatalog(media_storage, artifact_storage or media_storage, MEDIA_URLS)


class FakeApiStorage:
    async def upload_bytes(self, data: bytes | BinaryIO, key: str, content_type: str) -> str:
        return f"etag:{key}"

    async def delete(self, key: str) -> None:
        pass

    async def exists(self, key: str) -> bool:
        return True


async def test_list_images_empty(mock_storage: MagicMock) -> None:
    page = await catalog(mock_storage).list_images(limit=20, offset=0)

    assert page.images == []
    assert page.total == 0
    assert page.has_more is False


async def test_list_images_with_pagination(run_sync_seed, mock_storage: MagicMock) -> None:
    await run_sync_seed(lambda session: [create_image(session=session) for _ in range(3)])

    page = await catalog(mock_storage).list_images(limit=2, offset=0)

    assert len(page.images) == 2
    assert page.total == 3
    assert page.has_more is True


async def test_list_images_filters_by_dataset(run_sync_seed, mock_storage: MagicMock) -> None:
    def seed(session) -> None:
        create_image(session=session, dataset="cats")
        create_image(session=session, dataset="dogs")

    await run_sync_seed(seed)

    page = await catalog(mock_storage).list_images(
        limit=20,
        offset=0,
        dataset="cats",
    )

    assert page.total == 1
    assert page.images[0].dataset == "cats"


async def test_list_images_sorts_oldest(run_sync_seed, mock_storage: MagicMock) -> None:
    def seed(session) -> tuple[int, int]:
        first = create_image(session=session)
        second = create_image(session=session)
        return first.id, second.id

    first_id, second_id = await run_sync_seed(seed)

    page = await catalog(mock_storage).list_images(
        limit=20,
        offset=0,
        sort="oldest",
    )

    assert [image.id for image in page.images] == [first_id, second_id]


@pytest.mark.parametrize(
    ("processing_overrides", "expected_status"),
    [
        ({}, "pending"),
        ({"embed_status": ProcessingStatus.DONE}, "done"),
        ({"ocr_status": ProcessingStatus.FAILED}, "failed"),
        ({"embed_status": ProcessingStatus.RUNNING}, "embedding"),
        ({"caption_status": ProcessingStatus.RUNNING}, "annotating"),
    ],
)
async def test_list_images_projects_status(
    run_sync_seed,
    mock_storage: MagicMock,
    processing_overrides: dict[str, ProcessingStatus],
    expected_status: str,
) -> None:
    def seed(session) -> None:
        image = create_image(session=session)
        create_processing(session=session, image=image, **processing_overrides)

    await run_sync_seed(seed)

    page = await catalog(mock_storage).list_images(limit=20, offset=0)

    assert page.images[0].status == expected_status


async def test_list_images_filters_by_projected_status(
    run_sync_seed,
    mock_storage: MagicMock,
) -> None:
    def seed(session) -> int:
        done = create_image(session=session)
        create_processing(session=session, image=done, embed_status=ProcessingStatus.DONE)
        failed = create_image(session=session)
        create_processing(session=session, image=failed, caption_status=ProcessingStatus.FAILED)
        return failed.id

    failed_id = await run_sync_seed(seed)

    page = await catalog(mock_storage).list_images(
        limit=20,
        offset=0,
        status="failed",
    )

    assert page.total == 1
    assert page.images[0].id == failed_id


async def test_get_image_with_annotation(run_sync_seed, mock_storage: MagicMock) -> None:
    def seed(session) -> int:
        image = create_image(session=session)
        create_annotation(session=session, image=image, caption_text="A cat", ocr_text="LOL")
        return image.id

    image_id = await run_sync_seed(seed)

    result = await catalog(mock_storage).get_image(image_id)

    assert result.id == image_id
    assert result.caption == "A cat"
    assert result.ocr_text == "LOL"


async def test_get_image_without_annotation(run_sync_seed, mock_storage: MagicMock) -> None:
    image_id = await run_sync_seed(lambda session: create_image(session=session).id)

    result = await catalog(mock_storage).get_image(image_id)

    assert result.id == image_id
    assert result.caption is None
    assert result.ocr_text is None


async def test_get_image_missing(mock_storage: MagicMock) -> None:
    with pytest.raises(ImageCatalogNotFoundError):
        await catalog(mock_storage).get_image(999999)


async def test_public_media_url_attached_when_storage_key_exists(
    run_sync_seed,
    mock_storage: MagicMock,
) -> None:
    image_id = await run_sync_seed(
        lambda session: create_image(session=session, s3_key="images/test/example.jpg").id
    )

    result = await catalog(mock_storage).get_image(image_id)

    assert result.url == "https://assets.mimeme.dev/images/test/example.jpg"


async def test_image_catalog_encodes_public_media_key(run_sync_seed) -> None:
    storage = FakeApiStorage()
    image_id = await run_sync_seed(
        lambda session: create_image(session=session, s3_key="images/test/my example.jpg").id
    )

    result = await catalog(storage).get_image(image_id)

    assert result.url == "https://assets.mimeme.dev/images/test/my%20example.jpg"


async def test_delete_image_removes_database_rows_and_storage_artifacts(
    async_db_session: AsyncSession,
    run_sync_seed,
) -> None:
    media_storage = MagicMock()
    media_storage.delete = AsyncMock()
    artifact_storage = MagicMock()
    artifact_storage.delete = AsyncMock()

    def seed(session) -> int:
        image = create_image(session=session, s3_key="images/test/example.jpg")
        create_processing(
            session=session,
            image=image,
            embed_s3_key="embeddings/test/example.npy",
        )
        create_annotation(session=session, image=image)
        create_artifact(session=session, image=image)
        return image.id

    image_id = await run_sync_seed(seed)

    await catalog(media_storage, artifact_storage).delete_image(image_id)

    assert await async_db_session.get(Image, image_id) is None
    assert (
        await async_db_session.scalar(select(Processing).where(Processing.image_id == image_id))
    ) is None
    assert (
        await async_db_session.scalar(select(Annotation).where(Annotation.image_id == image_id))
    ) is None
    assert (
        await async_db_session.scalar(select(Artifact).where(Artifact.image_id == image_id))
    ) is None
    media_storage.delete.assert_awaited_once_with("images/test/example.jpg")
    assert artifact_storage.delete.await_args_list[0].args == ("embeddings/test/example.npy",)
    assert artifact_storage.delete.await_args_list[1].args == ("embeddings/test/example_text.npy",)


async def _desired_generation(session: AsyncSession) -> int:
    value = await session.scalar(
        select(SearchIndexState.desired_generation).where(SearchIndexState.id == 1)
    )
    assert value is not None
    return value


async def test_delete_searchable_image_increments_generation(
    async_db_session: AsyncSession,
    run_sync_seed,
) -> None:
    media_storage = MagicMock()
    media_storage.delete = AsyncMock()
    artifact_storage = MagicMock()
    artifact_storage.delete = AsyncMock()

    def seed(session) -> int:
        create_search_index_state(session=session, desired_generation=5, active_generation=5)
        image = create_image(session=session, s3_key="images/test/x.jpg")
        create_processing(
            session=session,
            image=image,
            embed_status=ProcessingStatus.DONE,
            embed_s3_key="embeddings/test/x.npy",
        )
        return image.id

    image_id = await run_sync_seed(seed)

    await catalog(media_storage, artifact_storage).delete_image(image_id)

    assert await async_db_session.get(Image, image_id) is None
    assert await _desired_generation(async_db_session) == 6


@pytest.mark.parametrize(
    "embed_status",
    [ProcessingStatus.PENDING, ProcessingStatus.DONE],
    ids=["pending", "done-no-key"],
)
async def test_delete_non_searchable_image_does_not_increment(
    async_db_session: AsyncSession,
    run_sync_seed,
    embed_status: ProcessingStatus,
) -> None:
    media_storage = MagicMock()
    media_storage.delete = AsyncMock()

    def seed(session) -> int:
        create_search_index_state(session=session, desired_generation=5, active_generation=5)
        image = create_image(session=session, s3_key="images/test/x.jpg")
        create_processing(
            session=session, image=image, embed_status=embed_status, embed_s3_key=None
        )
        return image.id

    image_id = await run_sync_seed(seed)

    await catalog(media_storage).delete_image(image_id)

    assert await async_db_session.get(Image, image_id) is None
    assert await _desired_generation(async_db_session) == 5


async def test_delete_searchable_image_increments_even_when_storage_delete_fails(
    async_db_session: AsyncSession,
    run_sync_seed,
) -> None:
    storage = MagicMock()
    storage.delete = AsyncMock(side_effect=RuntimeError("storage unavailable"))

    def seed(session) -> int:
        create_search_index_state(session=session, desired_generation=2, active_generation=2)
        image = create_image(session=session, s3_key="images/test/x.jpg")
        create_processing(
            session=session,
            image=image,
            embed_status=ProcessingStatus.DONE,
            embed_s3_key="embeddings/test/x.npy",
        )
        return image.id

    image_id = await run_sync_seed(seed)

    await catalog(storage).delete_image(image_id)

    assert await async_db_session.get(Image, image_id) is None
    assert await _desired_generation(async_db_session) == 3


async def test_delete_image_missing(mock_storage: MagicMock) -> None:
    with pytest.raises(ImageCatalogNotFoundError):
        await catalog(mock_storage).delete_image(999999)


async def test_delete_image_storage_failure_does_not_block_database_deletion(
    async_db_session: AsyncSession,
    run_sync_seed,
) -> None:
    storage = MagicMock()
    storage.delete = AsyncMock(side_effect=RuntimeError("storage unavailable"))
    image_id = await run_sync_seed(
        lambda session: create_image(session=session, s3_key="images/test/example.jpg").id
    )

    await catalog(storage).delete_image(image_id)

    assert await async_db_session.get(Image, image_id) is None
