from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError
from temporalio.contrib.pydantic import pydantic_data_converter

from mimeme.db.schema import IngestInputKind
from mimeme.ingest import rule
from mimeme.ingest.model import (
    Input,
    ItemRef,
    RemoteUrl,
    Result,
    Source,
    Staged,
    Submission,
    WorkflowInput,
    restore,
)


class TestSourceUnion:
    def test_discriminates_on_kind(self) -> None:
        adapter = TypeAdapter(Source)
        remote = adapter.validate_python({"kind": "remote_image_url", "url": "https://a/1.jpg"})
        staged = adapter.validate_python({"kind": "staged_upload", "artifact_key": "s/x.png"})
        assert isinstance(remote, RemoteUrl)
        assert isinstance(staged, Staged)

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            RemoteUrl(url="https://a", extra="no")  # type: ignore[call-arg]

    def test_restore_roundtrip(self) -> None:
        assert restore(
            kind=IngestInputKind.REMOTE_IMAGE_URL, url="https://a", artifact_key=None
        ) == RemoteUrl(url="https://a")
        assert restore(kind=IngestInputKind.STAGED_UPLOAD, url=None, artifact_key="k") == Staged(
            artifact_key="k"
        )

    def test_restore_requires_payload(self) -> None:
        with pytest.raises(ValueError):
            restore(kind=IngestInputKind.REMOTE_IMAGE_URL, url=None, artifact_key=None)


class TestSubmission:
    def test_bounds_url_count(self) -> None:
        with pytest.raises(ValidationError):
            Submission(urls=[])
        with pytest.raises(ValidationError):
            Submission(urls=[RemoteUrl(url=f"https://a/{i}") for i in range(101)])
        assert Submission(urls=[RemoteUrl(url="https://a/1")]).dataset is None


class TestTemporalConversion:
    async def test_workflow_input_and_result_convert(self) -> None:
        value = WorkflowInput(
            job_id="ingest-1",
            dataset="d",
            items=[ItemRef(item_id=1, source=RemoteUrl(url="https://a/1"))],
        )
        payload = await pydantic_data_converter.encode([value])
        [decoded] = await pydantic_data_converter.decode(payload, [WorkflowInput])
        assert decoded == value

    async def test_result_converts(self) -> None:
        value = Result(item_id=3, outcome="duplicate", image_id=9)
        payload = await pydantic_data_converter.encode([value])
        [decoded] = await pydantic_data_converter.decode(payload, [Result])
        assert decoded == value

    async def test_input_rejects_unknown_field_on_decode(self) -> None:
        with pytest.raises(ValidationError):
            Input.model_validate(
                {
                    "job_id": "j",
                    "item_id": 1,
                    "source": {"kind": "remote_image_url", "url": "https://a"},
                    "surprise": True,
                }
            )


class TestRule:
    def test_workflow_id(self) -> None:
        assert rule.workflow_id("ingest-abc") == "ingest-v2-ingest-abc"

    def test_temporal_names(self) -> None:
        assert rule.WORKFLOW == "mimeme.ingest.v2"
        assert rule.ITEM_ACTIVITY == "mimeme.ingest.item.v2"
        assert rule.FINISH_ACTIVITY == "mimeme.ingest.finish.v2"
        assert rule.TASK_QUEUE == "mimeme-v2"

    def test_canonical_media_key_is_content_addressed(self) -> None:
        key = rule.canonical_media_key(sha256="ab" * 32, dataset="memes", image_format="PNG")
        assert key == f"images/memes/ab/ab/{'ab' * 32}.png"
        assert rule.canonical_media_key(
            sha256="cd" * 32, dataset=None, image_format=None
        ).startswith("images/api-ingested/")

    @pytest.mark.parametrize(
        ("status", "terminal"),
        [(404, True), (403, True), (400, True), (408, False), (429, False), (425, False)],
    )
    def test_http_status_classification(self, status: int, terminal: bool) -> None:
        assert rule.is_terminal_http_status(status) is terminal

    def test_staging_keys(self) -> None:
        assert rule.staging_key(7) == "uploads/ingest-staging/7"
        assert rule.upload_staging_key("cat.PNG", token="tok").startswith("uploads/staging/tok")
        assert rule.upload_staging_key("cat.PNG", token="tok").endswith(".png")

    def test_progress_percent(self) -> None:
        assert rule.progress_percent(1, 4) == 25.0
        assert rule.progress_percent(0, 0) == 0.0
