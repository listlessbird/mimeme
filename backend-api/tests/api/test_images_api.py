from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from shared.models import Annotation, IngestURL, ORMImage, Processing, ProcessingStatus


def test_ingest_images_deduplicates_urls_and_starts_workflow(
    client, admin_headers, fake_temporal_client, app
) -> None:
    payload = {
        "urls": [
            "https://example.com/m1.jpg",
            "https://example.com/m1.jpg",
            "https://example.com/m2.jpg",
        ],
        "dataset": "memes",
        "tags": ["funny"],
    }
    response = client.post("/images", json=payload, headers=admin_headers)
    assert response.status_code == 202

    body = response.json()
    assert body["queued"] == 2
    assert body["duplicates"] == 1
    assert body["job_id"].startswith("ingest-")

    assert len(fake_temporal_client.started_workflows) == 1
    _, kwargs = fake_temporal_client.started_workflows[0]
    assert kwargs["id"].startswith("ingest-workflow-ingest-")

    sf: sessionmaker = app.state.session_factory
    with sf() as db:
        urls = db.query(IngestURL).all()
        assert len(urls) == 2


def test_list_images_filters_done_and_generates_presigned_url(
    client, admin_headers, app
) -> None:
    sf: sessionmaker = app.state.session_factory
    with sf() as db:
        done_img = ORMImage(
            id=1,
            sha256="sha-done",
            dataset="ds",
            s3_key="img/done.jpg",
            width=640,
            height=480,
        )
        pending_img = ORMImage(
            id=2,
            sha256="sha-pending",
            dataset="ds",
            s3_key="img/pending.jpg",
            width=100,
            height=100,
        )
        db.add_all([done_img, pending_img])
        db.add(
            Processing(
                image_id=1,
                ocr_status=ProcessingStatus.DONE,
                caption_status=ProcessingStatus.DONE,
                embed_status=ProcessingStatus.DONE,
            )
        )
        db.add(
            Processing(
                image_id=2,
                ocr_status=ProcessingStatus.PENDING,
                caption_status=ProcessingStatus.PENDING,
                embed_status=ProcessingStatus.PENDING,
            )
        )
        db.commit()

    response = client.get(
        "/images",
        params={"status": "done", "dataset": "ds"},
        headers=admin_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert len(body["images"]) == 1
    assert body["images"][0]["id"] == 1
    assert body["images"][0]["status"] == "done"
    assert body["images"][0]["url"] is not None


def test_get_and_delete_image_lifecycle(client, admin_headers, app, fake_storage) -> None:
    sf: sessionmaker = app.state.session_factory
    with sf() as db:
        img = ORMImage(id=7, sha256="sha-7", s3_key="img/7.jpg", dataset="x")
        db.add(img)
        db.add(
            Processing(
                image_id=7,
                ocr_status=ProcessingStatus.DONE,
                caption_status=ProcessingStatus.DONE,
                embed_status=ProcessingStatus.DONE,
                embed_s3_key="emb/7.npy",
            )
        )
        db.add(Annotation(image_id=7, caption_text="caption", ocr_text="ocr"))
        db.commit()

    get_resp = client.get("/images/7", headers=admin_headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == 7

    del_resp = client.delete("/images/7", headers=admin_headers)
    assert del_resp.status_code == 204

    with sf() as db:
        assert db.query(ORMImage).filter_by(id=7).first() is None
        assert db.query(Processing).filter_by(image_id=7).first() is None
        assert db.query(Annotation).filter_by(image_id=7).first() is None

    assert "img/7.jpg" in fake_storage.delete_calls
    assert "emb/7.npy" in fake_storage.delete_calls
    assert "emb/7_text.npy" in fake_storage.delete_calls

