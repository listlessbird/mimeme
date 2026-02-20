from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from shared.models import Job, JobStatus, JobType


def test_trigger_rebuild_index_starts_temporal_workflow(
    client, admin_headers, fake_temporal_client, app
) -> None:
    response = client.post("/jobs/rebuild-index", json={"force": True}, headers=admin_headers)
    assert response.status_code == 202
    body = response.json()
    assert body["id"].startswith("rebuild-")
    assert body["type"] == "rebuild_index"
    assert body["status"] == "PENDING"

    assert len(fake_temporal_client.started_workflows) == 1
    _, kwargs = fake_temporal_client.started_workflows[0]
    assert kwargs["id"].startswith("rebuild-workflow-rebuild-")

    sf: sessionmaker = app.state.session_factory
    with sf() as db:
        job = db.query(Job).filter_by(id=body["id"]).first()
        assert job is not None
        assert job.workflow_id is not None


def test_get_job_returns_raw_result_on_invalid_json(client, admin_headers, app) -> None:
    sf: sessionmaker = app.state.session_factory
    with sf() as db:
        db.add(
            Job(
                id="job-invalid-json",
                type=JobType.INGEST,
                status=JobStatus.RUNNING,
                progress=20.0,
                result="not-json",
            )
        )
        db.commit()

    response = client.get("/jobs/job-invalid-json", headers=admin_headers)
    assert response.status_code == 200
    assert response.json()["result"] == {"raw": "not-json"}


def test_cancel_job_rejects_completed_job(client, admin_headers, app) -> None:
    sf: sessionmaker = app.state.session_factory
    with sf() as db:
        db.add(
            Job(
                id="job-complete",
                type=JobType.INGEST,
                status=JobStatus.COMPLETED,
                progress=100.0,
            )
        )
        db.commit()

    response = client.delete("/jobs/job-complete", headers=admin_headers)
    assert response.status_code == 400
    assert "Cannot cancel completed job" in response.json()["detail"]


def test_cancel_job_marks_cancelled_and_calls_temporal(
    client, admin_headers, fake_temporal_client, app
) -> None:
    sf: sessionmaker = app.state.session_factory
    with sf() as db:
        db.add(
            Job(
                id="job-running",
                type=JobType.REBUILD_INDEX,
                status=JobStatus.RUNNING,
                progress=50.0,
                workflow_id="wf-job-running",
            )
        )
        db.commit()

    response = client.delete("/jobs/job-running", headers=admin_headers)
    assert response.status_code == 204
    assert fake_temporal_client.cancelled_workflows == ["wf-job-running"]

    with sf() as db:
        job = db.query(Job).filter_by(id="job-running").first()
        assert job is not None
        assert job.status == JobStatus.CANCELLED

