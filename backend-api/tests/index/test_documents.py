from __future__ import annotations

from sqlalchemy.orm import Session
from tests.factories import (
    create_annotation,
    create_image,
    create_ingest_url,
    create_ingestion_source,
    create_job,
    create_processing,
    create_search_index_state,
    create_source_item,
)
from tests.job.conftest import SavepointDb
from tests.support.storage import Memory

from mimeme import storage
from mimeme.db.schema import ProcessingStatus
from mimeme.index import documents
from mimeme.job import ops as job_ops
from mimeme.search.document import SearchDocument

_MODEL = "test/embed"


async def test_document_artifact_round_trips_canonical_jsonl() -> None:
    artifacts = Memory()
    values = [
        SearchDocument(image_id=1, titles=("First",)),
        SearchDocument(image_id=2, tags=("reaction", "template")),
    ]

    descriptor = await documents.publish(artifacts, version="v2-g1-test", documents=values)

    assert descriptor.key == "indexes/v2-g1-test/documents.jsonl.zst"
    assert descriptor.count == 2
    assert descriptor.length > 0
    assert await documents.verify(artifacts, descriptor) == values


async def test_document_artifact_rejects_a_wrong_content_hash() -> None:
    artifacts = Memory()
    descriptor = await documents.publish(
        artifacts,
        version="v2-g1-test",
        documents=[SearchDocument(image_id=1, titles=("First",))],
    )

    corrupt = descriptor.model_copy(update={"content_sha256": "0" * 64})

    try:
        await documents.verify(artifacts, corrupt)
    except ValueError as exc:
        assert "content hash mismatch" in str(exc)
    else:
        raise AssertionError("a wrong document content hash was accepted")


async def test_document_artifact_rejects_corrupt_compressed_bytes() -> None:
    artifacts = Memory()
    descriptor = await documents.publish(
        artifacts,
        version="v2-g1-test",
        documents=[SearchDocument(image_id=1)],
    )
    await artifacts.put_bytes(
        storage.Object(descriptor.key), b"not zstd", content_type="application/zstd"
    )

    try:
        await documents.verify(
            artifacts,
            descriptor.model_copy(
                update={
                    "length": len(b"not zstd"),
                    "sha256": storage.Checksum.of(b"not zstd").value,
                }
            ),
        )
    except ValueError as exc:
        assert "corrupt" in str(exc)
    else:
        raise AssertionError("corrupt compressed document bytes were accepted")


def _embedded(session: Session, *, key: str):  # noqa: ANN202
    image = create_image(session=session)
    processing = create_processing(session=session, image=image)
    processing.embed_status = ProcessingStatus.DONE
    processing.embed_model = _MODEL
    processing.embed_dim = 4
    processing.embed_s3_key = key
    processing.embed_text_present = False
    session.flush()
    return image


async def test_capture_projects_one_document_for_each_eligible_image(
    index_db: SavepointDb, run_sync_seed
) -> None:  # noqa: ANN001
    def seed(session: Session) -> tuple[int, int]:
        job = create_job(session=session)
        source = create_ingestion_source(session=session)
        bare = _embedded(session, key="embeddings/bare.npy")
        linked = _embedded(session, key="embeddings/linked.npy")
        create_annotation(
            session=session,
            image=linked,
            caption_text="  a caption ",
            ocr_text="TOP\nTEXT",
        )
        first = create_source_item(
            session=session,
            source=source,
            source_id=source.id,
            title="Alias B",
            known_facts={
                "title": "Canonical",
                "tags": ["reaction", "same"],
                "categories": ["People"],
                "types": ["Image macro"],
                "origin": "Stock photo",
                "year": "2017",
                "description": "A distracted man.",
            },
        )
        second = create_source_item(
            session=session,
            source=source,
            source_id=source.id,
            title="Alias A",
            known_facts={"title": "Canonical", "tags": ["same", "template"]},
        )
        create_ingest_url(
            session=session,
            job=job,
            job_id=job.id,
            image=linked,
            image_id=linked.id,
            source_id=source.id,
            source_item_id=first.id,
            status=ProcessingStatus.DONE,
        )
        create_ingest_url(
            session=session,
            job=job,
            job_id=job.id,
            image=linked,
            image_id=linked.id,
            source_id=source.id,
            source_item_id=second.id,
            status=ProcessingStatus.DONE,
        )
        return bare.id, linked.id

    bare_id, linked_id = await run_sync_seed(seed)
    async with index_db.read_session() as session:
        snapshot = await documents.capture(session, model=_MODEL, target_generation=4)

    assert [embedding.image_id for embedding in snapshot.embeddings] == [bare_id, linked_id]
    assert [value.image_id for value in snapshot.documents] == [bare_id, linked_id]
    assert snapshot.documents[0].titles == ()
    linked = snapshot.documents[1]
    assert linked.titles == ("Alias A", "Alias B", "Canonical")
    assert linked.tags == ("reaction", "same", "template")
    assert linked.categories == ("People",)
    assert linked.types == ("Image macro",)
    assert linked.origins == ("Stock photo",)
    assert linked.years == ("2017",)
    assert linked.captions == ("a caption",)
    assert linked.ocr_texts == ("TOP TEXT",)
    assert linked.descriptions == ("A distracted man.",)


async def test_capture_ignores_sources_linked_only_to_ineligible_images(
    index_db: SavepointDb, run_sync_seed
) -> None:  # noqa: ANN001
    def seed(session: Session) -> int:
        job = create_job(session=session)
        ingestion_source = create_ingestion_source(session=session)
        eligible = _embedded(session, key="embeddings/eligible.npy")
        ineligible = create_image(session=session)
        source = create_source_item(
            session=session,
            source=ingestion_source,
            source_id=ingestion_source.id,
            title="Not searchable",
        )
        create_ingest_url(
            session=session,
            job=job,
            job_id=job.id,
            image=ineligible,
            image_id=ineligible.id,
            source_id=ingestion_source.id,
            source_item_id=source.id,
            status=ProcessingStatus.DONE,
        )
        return eligible.id

    eligible_id = await run_sync_seed(seed)
    async with index_db.read_session() as session:
        snapshot = await documents.capture(session, model=_MODEL, target_generation=5)

    assert [value.image_id for value in snapshot.documents] == [eligible_id]


async def test_fact_change_after_capture_stays_dirty_for_the_next_generation(
    index_db: SavepointDb, run_sync_seed
) -> None:  # noqa: ANN001
    def seed(session: Session) -> int:
        create_search_index_state(session=session, desired_generation=1, active_generation=1)
        image = _embedded(session, key="embeddings/ready.npy")
        create_annotation(session=session, image=image, caption_text="before", ocr_text="")
        return image.id

    image_id = await run_sync_seed(seed)
    async with index_db.read_session() as session:
        snapshot = await documents.capture(session, model=_MODEL, target_generation=1)

    await job_ops.save_annotations(
        index_db,
        image_id=image_id,
        caption="after",
        caption_model="m",
        ocr_text="",
        ocr_model="m",
    )

    assert snapshot.documents[0].captions == ("before",)
    assert (await job_ops.index_status(index_db)).view.desired_generation == 2
