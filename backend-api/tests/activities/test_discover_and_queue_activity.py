"""Tests for discover_and_queue_activity (issue 03, slice 4).

The DB-touching heart of a Source run: parse the fetched responses, run the pure
Source-item dedup, record *every* distinct discovered item as a ``source_item``
(so future runs remember it), and queue only the genuinely-new ones as an INGEST
Job whose ``ingest_urls`` carry Source provenance.

The pure parse/dedup decisions are covered in tests/domain/; here we assert the
persistence and queueing behavior, plus idempotency under retry.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy.orm import Session
from temporalio.testing import ActivityEnvironment

from mimeme.activities.ingestion.activities import discover_and_queue_activity
from mimeme.activities.ingestion.models import DiscoverAndQueueInput
from mimeme.db.schema import IngestURL, Job, JobType, SourceItem
from tests.factories import create_ingestion_source, create_source_item, create_source_run


@pytest.fixture()
def activity_env() -> ActivityEnvironment:
    return ActivityEnvironment()


def _meme(post_id: str, *, nsfw: bool = False, url: str | None = None) -> dict[str, Any]:
    return {
        "postLink": f"https://redd.it/{post_id}",
        "url": url or f"https://i.redd.it/{post_id}.png",
        "title": f"title {post_id}",
        "author": "author",
        "ups": 1,
        "subreddit": "memes",
        "preview": [],
        "nsfw": nsfw,
        "spoiler": False,
    }


def _response(*memes: dict[str, Any]) -> dict[str, Any]:
    return {"count": len(memes), "memes": list(memes)}


def _setup(session: Session) -> tuple[Any, Any]:
    source = create_ingestion_source(session=session, adapter_key="meme_api", dataset="memes-ds")
    run = create_source_run(session=session, source=source, source_id=source.id)
    return source, run


def _input(source: Any, run: Any, *responses: dict[str, Any]) -> DiscoverAndQueueInput:
    return DiscoverAndQueueInput(
        source_id=source.id,
        source_run_id=run.id,
        adapter_key="meme_api",
        dataset=source.dataset,
        raw_responses=list(responses),
    )


def _items(session: Session, source_id: int) -> list[SourceItem]:
    return session.query(SourceItem).filter(SourceItem.source_id == source_id).all()


def _urls(session: Session, source_run_id: int) -> list[IngestURL]:
    return session.query(IngestURL).filter(IngestURL.source_run_id == source_run_id).all()


@pytest.mark.usefixtures("_patch_session_scope")
class TestDiscoverAndQueueActivity:
    def test_new_items_recorded_and_queued_with_provenance(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        source, run = _setup(db_session)

        result = activity_env.run(
            discover_and_queue_activity,
            _input(source, run, _response(_meme("aaa"), _meme("bbb"))),
        )

        assert result.discovered == 2
        assert result.queued == 2
        assert result.ingest_job_id is not None

        items = _items(db_session, source.id)
        assert {i.external_item_id for i in items} == {"aaa", "bbb"}
        assert all(i.last_source_run_id == run.id for i in items)

        urls = _urls(db_session, run.id)
        assert {u.url for u in urls} == {
            "https://i.redd.it/aaa.png",
            "https://i.redd.it/bbb.png",
        }
        source_item_ids = {i.id for i in items}
        for url in urls:
            assert url.source_id == source.id
            assert url.source_run_id == run.id
            assert url.source_item_id in source_item_ids

        db_session.refresh(run)
        assert run.ingest_job_id == result.ingest_job_id
        job = db_session.get(Job, result.ingest_job_id)
        assert job is not None
        assert job.type == JobType.INGEST

    def test_already_seen_items_are_recorded_but_not_requeued(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        source, run = _setup(db_session)
        old_seen_at = datetime(2024, 1, 1, tzinfo=UTC)
        create_source_item(
            session=db_session,
            source=source,
            source_id=source.id,
            external_item_id="aaa",
            last_seen_at=old_seen_at,
        )
        create_source_item(
            session=db_session,
            source=source,
            source_id=source.id,
            external_item_id="ccc",
            last_seen_at=old_seen_at,
        )

        result = activity_env.run(
            discover_and_queue_activity,
            _input(source, run, _response(_meme("aaa"), _meme("bbb"), _meme("ccc"))),
        )

        assert result.discovered == 3
        assert result.queued == 1

        items = _items(db_session, source.id)
        assert {i.external_item_id for i in items} == {"aaa", "bbb", "ccc"}
        seen_items = [i for i in items if i.external_item_id in {"aaa", "ccc"}]
        assert all(i.last_source_run_id == run.id for i in seen_items)
        assert all(i.last_seen_at > old_seen_at for i in seen_items)
        inserted = next(i for i in items if i.external_item_id == "bbb")
        assert inserted.first_seen_at == inserted.last_seen_at

        urls = _urls(db_session, run.id)
        assert {u.url for u in urls} == {"https://i.redd.it/bbb.png"}

    def test_within_run_duplicates_collapse(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        source, run = _setup(db_session)

        result = activity_env.run(
            discover_and_queue_activity,
            _input(source, run, _response(_meme("aaa"), _meme("aaa"), _meme("bbb"))),
        )

        assert result.discovered == 2
        assert result.queued == 2
        assert len(_items(db_session, source.id)) == 2
        assert len(_urls(db_session, run.id)) == 2

    def test_dropped_items_are_neither_recorded_nor_queued(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        source, run = _setup(db_session)

        result = activity_env.run(
            discover_and_queue_activity,
            _input(
                source,
                run,
                _response(
                    _meme("keep"),
                    _meme("nope", nsfw=True),
                    _meme("animated", url="https://i.redd.it/animated.gif"),
                ),
            ),
        )

        assert result.discovered == 1
        assert result.queued == 1
        assert {i.external_item_id for i in _items(db_session, source.id)} == {"keep"}

    def test_no_new_items_creates_no_job(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        source, run = _setup(db_session)
        create_source_item(
            session=db_session, source=source, source_id=source.id, external_item_id="aaa"
        )

        result = activity_env.run(
            discover_and_queue_activity,
            _input(source, run, _response(_meme("aaa"))),
        )

        assert result.discovered == 1
        assert result.queued == 0
        assert result.ingest_job_id is None

        db_session.refresh(run)
        assert run.ingest_job_id is None
        assert _urls(db_session, run.id) == []

    def test_retry_is_idempotent_and_does_not_double_queue(
        self, db_session: Session, activity_env: ActivityEnvironment
    ) -> None:
        source, run = _setup(db_session)
        payload = _input(source, run, _response(_meme("aaa"), _meme("bbb")))

        first = activity_env.run(discover_and_queue_activity, payload)
        second = activity_env.run(discover_and_queue_activity, payload)

        # The second run short-circuits on the already-linked job.
        assert second.ingest_job_id == first.ingest_job_id
        assert second.queued == 2
        assert len(_items(db_session, source.id)) == 2
        assert len(_urls(db_session, run.id)) == 2
