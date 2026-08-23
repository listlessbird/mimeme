from __future__ import annotations

from sqlalchemy.orm import Session

from mimeme import search
from mimeme.db.schema import (
    SearchEvalJudgment,
    SearchEvalPoolCandidate,
    SearchEvalQuery,
)
from mimeme.search_eval import service
from tests.factories import create_image


def _seed(session: Session) -> tuple[int, int, int]:
    relevant = create_image(session=session)
    irrelevant = create_image(session=session)
    query = SearchEvalQuery(
        text="when the deploy breaks on friday",
        intent="situation",
        source="human",
        status="active",
    )
    session.add(query)
    session.flush()
    for image, grade in ((relevant, 3), (irrelevant, 0)):
        session.add(SearchEvalPoolCandidate(query_id=query.id, image_id=image.id))
        session.add(
            SearchEvalJudgment(query_id=query.id, image_id=image.id, grade=grade, revision=1)
        )
    session.flush()
    return query.id, relevant.id, irrelevant.id


async def test_run_lifecycle_persists_rankings_and_scores(
    eval_db,
    run_sync_seed,
) -> None:
    query_id, relevant_id, irrelevant_id = await run_sync_seed(_seed)
    created = await service.create_run(
        eval_db,
        run_id="run1",
        mode="hybrid",
        workflow_id="search-eval-v1-run1",
    )

    assert created.status == "queued"
    assert created.progress_total == 1

    prepared = await service.prepare_run(eval_db, created.id, index_version="index-v1")
    assert [query.id for query in prepared.queries] == [query_id]

    await service.record_query(
        eval_db,
        run_id=created.id,
        query_id=query_id,
        mode="hybrid",
        expected_index_version="index-v1",
        page=search.Page(
            query=prepared.queries[0].text,
            results=[
                search.Result(
                    id=relevant_id,
                    sha256="relevant",
                    score=0.9,
                    url=None,
                    caption=None,
                    ocr_text=None,
                    width=800,
                    height=600,
                ),
                search.Result(
                    id=irrelevant_id,
                    sha256="irrelevant",
                    score=0.8,
                    url=None,
                    caption=None,
                    ocr_text=None,
                    width=800,
                    height=600,
                ),
            ],
            total=2,
            limit=10,
            offset=0,
            has_more=False,
            search_time_ms=12,
            index_version="index-v1",
        ),
    )
    outcome = await service.score_run(eval_db, created.id)
    completed = await service.get_run(eval_db, created.id)

    assert outcome.status == "complete"
    assert completed.status == "complete"
    assert completed.progress_completed == 1
    assert completed.metrics is not None
    assert completed.metrics.ndcg_at_10 == 1
    assert completed.metrics.judged_at_10 == 1
