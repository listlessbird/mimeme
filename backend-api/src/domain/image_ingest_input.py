from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from shared.models.orm import IngestInputKind


class RemoteImageUrlInput(BaseModel, frozen=True):
    kind: Literal["remote_image_url"] = "remote_image_url"
    url: str = Field(min_length=1)


class StagedUploadInput(BaseModel, frozen=True):
    kind: Literal["staged_upload"] = "staged_upload"
    artifact_key: str = Field(min_length=1)


ImageIngestInput = Annotated[
    RemoteImageUrlInput | StagedUploadInput,
    Field(discriminator="kind"),
]


def restore_image_ingest_input(
    *,
    kind: IngestInputKind,
    url: str | None,
    artifact_key: str | None,
) -> ImageIngestInput:
    match kind:
        case IngestInputKind.REMOTE_IMAGE_URL:
            if url is None:
                raise ValueError("remote image input is missing its URL")
            return RemoteImageUrlInput(url=url)
        case IngestInputKind.STAGED_UPLOAD:
            if artifact_key is None:
                raise ValueError("staged upload input is missing its artifact key")
            return StagedUploadInput(artifact_key=artifact_key)
