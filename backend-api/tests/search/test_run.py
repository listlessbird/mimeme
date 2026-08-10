from __future__ import annotations

from mimeme import search
from mimeme.search.run import Projection, run


class _Client:
    def __init__(self, batches: list[search.Batch]) -> None:
        self._batches = iter(batches)
        self.cursors: list[str | None] = []

    async def query(
        self, query: search.Query, *, count: int, cursor: str | None = None
    ) -> search.Batch:
        self.cursors.append(cursor)
        return next(self._batches)

    async def status(self) -> search.Status:
        return search.Status(ready=True, serving_version="v1")

    async def close(self) -> None:
        pass


class _Rows:
    def __init__(self, rows: list[Projection]) -> None:
        self._rows = {row.id: row for row in rows}
        self.requests: list[list[int]] = []

    async def fetch(self, image_ids: list[int]) -> dict[int, Projection]:
        self.requests.append(image_ids)
        return {image_id: self._rows[image_id] for image_id in image_ids if image_id in self._rows}


class _Urls:
    def resolve(self, key: str) -> str:
        return f"https://media.test/{key}"


def _row(image_id: int) -> Projection:
    return Projection(
        id=image_id,
        sha256=f"sha-{image_id}",
        media_key=f"images/{image_id}.jpg",
        caption=f"caption {image_id}",
        ocr_text=None,
        width=800,
        height=600,
    )


async def test_run_fetches_more_candidates_for_sparse_rows_and_preserves_rank() -> None:
    client = _Client(
        [
            search.Batch(
                candidates=[
                    search.Candidate(image_id=99, score=0.99),
                    search.Candidate(image_id=1, score=0.9),
                ],
                cursor="next",
                exhausted=False,
                version="v1",
            ),
            search.Batch(
                candidates=[
                    search.Candidate(image_id=2, score=0.8),
                    search.Candidate(image_id=3, score=0.7),
                    search.Candidate(image_id=4, score=0.6),
                ],
                exhausted=True,
                version="v1",
            ),
        ]
    )
    rows = _Rows([_row(1), _row(2), _row(3), _row(4)])

    page = await run(
        search.Query(text="cat", mode="hybrid", limit=2, offset=1),
        client=client,
        rows=rows,
        media_urls=_Urls(),
    )

    assert [result.id for result in page.results] == [2, 3]
    assert [result.score for result in page.results] == [0.8, 0.7]
    assert page.total == 3
    assert page.has_more is False
    assert page.index_version == "v1"
    assert client.cursors == [None, "next"]
    assert page.results[0].url == "https://media.test/images/2.jpg"


async def test_similar_search_excludes_the_query_image_before_total() -> None:
    client = _Client(
        [
            search.Batch(
                candidates=[
                    search.Candidate(image_id=1, score=1.0),
                    search.Candidate(image_id=2, score=0.8),
                ],
                exhausted=True,
                version="v1",
            )
        ]
    )

    page = await run(
        search.Query(similar_image_id=1, limit=2),
        client=client,
        rows=_Rows([_row(1), _row(2)]),
        media_urls=_Urls(),
    )

    assert [result.id for result in page.results] == [2]
    assert page.total == 1
    assert page.has_more is False
    assert page.query == "similar_to:1"


async def test_run_reports_more_when_a_full_page_has_an_unexhausted_batch() -> None:
    client = _Client(
        [
            search.Batch(
                candidates=[
                    search.Candidate(image_id=1, score=0.9),
                    search.Candidate(image_id=2, score=0.8),
                    search.Candidate(image_id=3, score=0.7),
                ],
                exhausted=False,
                cursor="next",
                version="v1",
            )
        ]
    )

    page = await run(
        search.Query(text="cat", mode="hybrid", limit=2),
        client=client,
        rows=_Rows([_row(1), _row(2), _row(3)]),
        media_urls=_Urls(),
    )

    assert [result.id for result in page.results] == [1, 2]
    assert page.has_more is True
