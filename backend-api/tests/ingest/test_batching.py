from __future__ import annotations

from types import SimpleNamespace

from mimeme import inference
from mimeme.ingest import run as ingest_run
from mimeme.ingest.model import Input, RemoteUrl


async def test_run_batch_sends_prepared_items_in_one_inference_call(monkeypatch) -> None:  # noqa: ANN001
    inputs = [
        Input(job_id="job", item_id=index, source=RemoteUrl(url=f"https://a/{index}.png"))
        for index in range(3)
    ]

    async def prepare(_env, item: Input):  # noqa: ANN001, ANN202
        return ingest_run._PendingEmbedding(
            input=item,
            image_id=item.item_id,
            media_key=f"images/{item.item_id}.png",
            text=f"text {item.item_id}",
            text_sha256=f"text-sha-{item.item_id}",
            sha256=f"image-sha-{item.item_id}",
            outcome="processed",
            duplicate_reason=None,
            duplicate_of_image_id=None,
            similar_image_id=None,
            phash_distance=None,
            timings={},
            started=0,
        )

    async def noop(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None

    class FakeInference:
        def __init__(self) -> None:
            self.calls: list[inference.Batch] = []

        async def embed(self, batch: inference.Batch) -> inference.BatchResult:
            self.calls.append(batch)
            return inference.BatchResult(
                items=[
                    inference.Ok(
                        embedding=inference.Embedding(
                            image_id=item.image_id,
                            image_embedding_key=f"image-{item.image_id}.npy",
                            text_embedding_key=f"text-{item.image_id}.npy",
                            model="siglip2",
                            dimension=4,
                        )
                    )
                    for item in batch.items
                ]
            )

    client = FakeInference()
    monkeypatch.setattr(ingest_run, "_prepare", prepare)
    monkeypatch.setattr(ingest_run.job_ops, "save_embedding", noop)
    monkeypatch.setattr(ingest_run.job_ops, "mark_item_done", noop)
    monkeypatch.setattr(ingest_run.job_ops, "record_stage", noop)

    results = await ingest_run.run_batch(SimpleNamespace(inference=client, db=None), inputs)

    assert len(client.calls) == 1
    assert [item.image_id for item in client.calls[0].items] == [0, 1, 2]
    assert [result.item_id for result in results] == [0, 1, 2]
