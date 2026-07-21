import pytest
from pydantic import TypeAdapter
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from tests.factories import create_job

from mimeme.db.schema import IngestInputKind, IngestURL
from mimeme.domain.image_ingest_input import (
    ImageIngestInput,
    RemoteImageUrlInput,
    StagedUploadInput,
)


def test_image_ingest_input_serialization_is_exhaustively_discriminated() -> None:
    adapter = TypeAdapter(ImageIngestInput)

    remote = adapter.validate_python(
        {"kind": "remote_image_url", "url": "https://example.com/meme.jpg"}
    )
    staged = adapter.validate_python(
        {"kind": "staged_upload", "artifact_key": "uploads/staging/abc.jpg"}
    )

    assert remote == RemoteImageUrlInput(url="https://example.com/meme.jpg")
    assert staged == StagedUploadInput(artifact_key="uploads/staging/abc.jpg")
    assert adapter.dump_python(remote) == {
        "kind": "remote_image_url",
        "url": "https://example.com/meme.jpg",
    }
    assert adapter.dump_python(staged) == {
        "kind": "staged_upload",
        "artifact_key": "uploads/staging/abc.jpg",
    }


@pytest.mark.parametrize(
    ("kind", "url", "artifact_key"),
    [
        (IngestInputKind.REMOTE_IMAGE_URL, None, None),
        (IngestInputKind.REMOTE_IMAGE_URL, "https://example.com/a.jpg", "uploads/a.jpg"),
        (IngestInputKind.STAGED_UPLOAD, "https://example.com/a.jpg", None),
        (IngestInputKind.STAGED_UPLOAD, None, None),
    ],
)
def test_persistence_rejects_contradictory_ingest_input_payloads(
    db_session: Session,
    kind: IngestInputKind,
    url: str | None,
    artifact_key: str | None,
) -> None:
    job = create_job(session=db_session)
    db_session.add(
        IngestURL(
            job_id=job.id,
            input_kind=kind,
            url=url,
            artifact_key=artifact_key,
        )
    )

    with pytest.raises(IntegrityError):
        db_session.flush()
