from __future__ import annotations

from httpx import AsyncClient

from shared.models import IngestStage, ProcessingStatus
from tests.factories import (
    create_image,
    create_ingest_url,
    create_ingestion_source,
    create_job,
    create_source_item,
    create_source_run,
)


async def test_ingestion_list_and_detail_use_async_browser(
    async_client: AsyncClient,
    run_sync_seed,
    _patch_async_domain_session_scope: None,
) -> None:
    def seed(session) -> tuple[int, int]:
        source = create_ingestion_source(session=session, name="r/memes")
        run = create_source_run(session=session, source=source)
        item = create_source_item(session=session, source=source)
        image = create_image(session=session, dataset="memes", s3_key="images/test/meme.jpg")
        job = create_job(session=session)
        attempt = create_ingest_url(
            session=session,
            job=job,
            job_id=job.id,
            source_id=source.id,
            source_run_id=run.id,
            source_item_id=item.id,
            status=ProcessingStatus.DONE,
            stage=IngestStage.COMPLETE,
            image_id=image.id,
        )
        return attempt.id, image.id

    attempt_id, image_id = await run_sync_seed(seed)

    list_response = await async_client.get("/ingestion", params={"view": "all"})
    assert list_response.status_code == 200
    rows = list_response.json()["rows"]
    assert len(rows) == 1
    assert rows[0]["ingest_url_id"] == attempt_id
    assert rows[0]["resolved_image_id"] == image_id
    assert rows[0]["thumbnail_url"] == "https://mock-s3/presigned"

    detail_response = await async_client.get(f"/ingestion/{attempt_id}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["ingest_url_id"] == attempt_id
    assert detail["image_id"] == image_id


async def test_source_item_and_run_item_routes_use_async_browser(
    async_client: AsyncClient,
    run_sync_seed,
    _patch_async_domain_session_scope: None,
) -> None:
    def seed(session) -> tuple[int, int, int]:
        source = create_ingestion_source(session=session)
        run = create_source_run(session=session, source=source)
        item = create_source_item(
            session=session,
            source=source,
            raw_metadata={"preview": "https://preview.example/item.jpg"},
        )
        image = create_image(session=session, s3_key="images/test/source-item.jpg")
        job = create_job(session=session)
        create_ingest_url(
            session=session,
            job=job,
            job_id=job.id,
            source_id=source.id,
            source_run_id=run.id,
            source_item_id=item.id,
            status=ProcessingStatus.DONE,
            image_id=image.id,
        )
        return source.id, run.id, item.id

    source_id, run_id, item_id = await run_sync_seed(seed)

    items_response = await async_client.get(f"/sources/{source_id}/items")
    assert items_response.status_code == 200
    items_body = items_response.json()
    assert items_body["total"] == 1
    assert items_body["items"][0]["id"] == item_id
    assert items_body["items"][0]["thumbnail_url"] == "https://mock-s3/presigned"

    run_items_response = await async_client.get(f"/sources/{source_id}/runs/{run_id}/items")
    assert run_items_response.status_code == 200
    run_items_body = run_items_response.json()
    assert run_items_body["total"] == 1
    assert run_items_body["items"][0]["source_item_id"] == item_id
    assert run_items_body["items"][0]["thumbnail_url"] == "https://mock-s3/presigned"
