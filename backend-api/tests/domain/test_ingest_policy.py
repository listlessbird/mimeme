from __future__ import annotations

from types import SimpleNamespace

from mimeme.domain.ingest_policy import IngestPolicy


def test_empty_url_list_progress_is_zero() -> None:
    policy = IngestPolicy(total=0)

    progress = policy.progress_state()
    completion = policy.completion("job-1")

    assert progress.progress == 0
    assert completion.total == 0
    assert completion.processed == 0


def test_success_updates_processed_count_and_progress() -> None:
    policy = IngestPolicy(total=2)

    done = policy.record_success(123)

    assert done.image_id == 123
    assert policy.progress_state().processed == 1
    assert policy.progress_state().progress == 50.0


def test_download_failure_updates_failed_count_and_compacts_error() -> None:
    policy = IngestPolicy(total=1)

    failure = policy.record_download_failure("x" * 600)

    assert policy.progress_state().failed == 1
    assert len(failure.error) == 500
    assert failure.error.endswith("...")


def test_duplicate_updates_duplicate_count_and_done_action() -> None:
    policy = IngestPolicy(total=4)

    done = policy.record_duplicate(42)

    assert done.image_id == 42
    assert policy.progress_state().duplicates == 1
    assert policy.progress_state().progress == 25.0


def test_embedding_text_composition_filters_empty_parts() -> None:
    policy = IngestPolicy(total=1)

    assert policy.compose_embedding_text("Caption", "OCR") == "Caption OCR"
    assert policy.compose_embedding_text("Caption", "") == "Caption"
    assert policy.compose_embedding_text("", "OCR") == "OCR"
    assert policy.compose_embedding_text("", "") == ""


def test_embedding_result_present_creates_save_decision() -> None:
    policy = IngestPolicy(total=1)

    decision = policy.embedding_decision(
        [
            SimpleNamespace(
                image_id=7,
                model="siglip",
                dimension=768,
                image_embedding_key="embeddings/1.npy",
            )
        ]
    )

    assert decision is not None
    assert decision.image_id == 7
    assert decision.image_embedding_key == "embeddings/1.npy"


def test_empty_embedding_results_have_no_save_decision() -> None:
    policy = IngestPolicy(total=1)

    assert policy.embedding_decision([]) is None


def test_exception_updates_failed_count_and_compacts_error() -> None:
    policy = IngestPolicy(total=2)

    failure = policy.record_exception(RuntimeError("line 1\nline 2"))

    assert policy.progress_state().failed == 1
    assert failure.error == "line 1 line 2"


def test_completion_payload_reflects_counts() -> None:
    policy = IngestPolicy(total=3)
    policy.record_success(1)
    policy.record_duplicate(2)
    policy.record_download_failure("HTTP 404")

    completion = policy.completion("job-1")

    assert completion.job_id == "job-1"
    assert completion.total == 3
    assert completion.processed == 1
    assert completion.failed == 1
    assert completion.duplicates == 1
