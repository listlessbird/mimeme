from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from mimeme import search
from mimeme.db.schema import (
    SearchEvalExperiment,
    SearchEvalJudgment,
    SearchEvalPoolCandidate,
    SearchEvalPoolSource,
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
        recipe_id="image_siglip_text",
        workflow_id="search-eval-v1-run1",
    )

    assert created.status == "queued"
    assert created.progress_total == 1
    assert created.recipe_id == "image_siglip_text"
    assert created.recipe.candidate_depth == 1000

    prepared = await service.prepare_run(eval_db, created.id, index_version="index-v1")
    assert [query.id for query in prepared.queries] == [query_id]
    assert prepared.recipe == created.recipe

    await service.record_query(
        eval_db,
        run_id=created.id,
        query_id=query_id,
        recipe_id="image_siglip_text",
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
    async with eval_db.read_session() as session:
        sources = (
            await session.scalars(
                select(SearchEvalPoolSource).where(SearchEvalPoolSource.query_id == query_id)
            )
        ).all()
        experiment = await session.get(SearchEvalExperiment, created.experiment_id)
    assert [source.recipe_id for source in sources] == ["image_siglip_text"] * 2
    assert experiment is not None and experiment.index_version == "index-v1"


async def test_grouped_experiment_freezes_one_snapshot_and_recipe_set(
    eval_db,
    run_sync_seed,
) -> None:
    await run_sync_seed(_seed)

    experiment = await service.create_experiment(
        eval_db,
        experiment_id="experiment1",
        runs=[
            ("image-run", "image_only", "workflow-image"),
            ("hybrid-run", "image_siglip_text", "workflow-hybrid"),
        ],
    )

    assert [run.recipe_id for run in experiment.runs] == [
        "image_only",
        "image_siglip_text",
    ]
    assert {run.snapshot_id for run in experiment.runs} == {experiment.snapshot_id}
    await service.prepare_run(eval_db, "image-run", index_version="index-v1")
    with pytest.raises(service.Conflict, match="experiment finished"):
        await service.prepare_run(eval_db, "hybrid-run", index_version="index-v2")
    await service.prepare_run(eval_db, "hybrid-run", index_version="index-v1")


async def test_bm25_experiment_freezes_lexical_settings(eval_db, run_sync_seed) -> None:
    await run_sync_seed(_seed)

    experiment = await service.create_experiment(
        eval_db,
        experiment_id="bm25-experiment",
        runs=[("bm25-run", "image_bm25", "workflow-bm25")],
    )
    definition = experiment.runs[0].recipe

    assert definition.id == "image_bm25"
    assert definition.bm25 == search.recipe.Bm25Settings(weights=(4, 4, 4, 2, 2, 2, 1))
    prepared = await service.prepare_run(eval_db, "bm25-run", index_version="index-v2")
    assert prepared.recipe == definition
    assert prepared.index_version == "index-v2"
