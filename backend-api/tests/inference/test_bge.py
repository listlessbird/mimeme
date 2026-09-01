from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from temporalio.contrib.pydantic import pydantic_data_converter

from mimeme.inference import bge
from mimeme.search.document import SearchDocument


def _export() -> bge.Export:
    return bge.Export(
        repo="project/bge-small-en-v1.5-onnx",
        revision="1" * 40,
        variant="model-int8.onnx",
        model_sha256="1" * 64,
        tokenizer_sha256="2" * 64,
        export_meta_sha256="3" * 64,
        opset=18,
    )


def test_document_renderer_is_labeled_deterministic_and_prefix_free() -> None:
    document = SearchDocument(
        image_id=7,
        titles=("First alias", "Second alias"),
        tags=("reaction",),
        captions=("A worried engineer",),
        ocr_texts=("DEPLOY FAILED " * 80,),
        categories=("Technology",),
        descriptions=("Used when a release goes badly",),
    )

    rendered = bge.render_document(document)

    assert rendered == (
        "Titles: First alias; Second alias\n"
        "Tags: reaction\n"
        "Captions: A worried engineer\n"
        f"OCR: {'DEPLOY FAILED ' * 80}\n"
        "Categories: Technology\n"
        "Descriptions: Used when a release goes badly"
    )
    assert not rendered.startswith(bge.QUERY_PREFIX)
    assert bge.render_document(document) == rendered


def test_document_renderer_omits_missing_fields() -> None:
    assert bge.render_document(SearchDocument(image_id=1, years=("2012",))) == "Years: 2012"
    assert bge.render_document(SearchDocument(image_id=2)) == ""


def test_query_renderer_uses_the_exact_versioned_prefix() -> None:
    assert bge.render_query("deploy failure") == (
        "Represent this sentence for searching relevant passages: deploy failure"
    )


def test_cls_pooling_normalizes_each_embedding() -> None:
    hidden = np.zeros((2, 3, bge.DIMENSION), dtype=np.float32)
    hidden[0, 0, :2] = [3, 4]
    hidden[1, 0, :2] = [5, 12]
    hidden[:, 1, 0] = 999

    vectors = bge.pool(hidden)

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, bge.DIMENSION)
    assert vectors[0, :2] == pytest.approx([0.6, 0.8])
    assert vectors[1, :2] == pytest.approx([5 / 13, 12 / 13])
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1, 1])


@pytest.mark.parametrize(
    "hidden",
    [
        np.zeros((1, 1, 12), dtype=np.float32),
        np.zeros((1, 1, bge.DIMENSION), dtype=np.float32),
        np.full((1, 1, bge.DIMENSION), np.nan, dtype=np.float32),
    ],
)
def test_pooling_rejects_incompatible_outputs(hidden: np.ndarray) -> None:
    with pytest.raises(ValueError):
        bge.pool(hidden)


async def test_encode_batch_roundtrips_and_rejects_moving_source_revisions() -> None:
    batch = bge.EncodeBatch(
        document_content_sha256="4" * 64,
        export=_export(),
        items=(bge.CorpusItem(image_id=1, text="Titles: Cat"),),
    )
    payloads = await pydantic_data_converter.encode([batch])
    (decoded,) = await pydantic_data_converter.decode(payloads, [bge.EncodeBatch])
    assert decoded == batch

    with pytest.raises(ValidationError):
        bge.Export.model_validate({**_export().model_dump(), "source_revision": "main"})
    with pytest.raises(ValidationError, match="image IDs must be unique"):
        bge.EncodeBatch(
            document_content_sha256="4" * 64,
            export=_export(),
            items=(
                bge.CorpusItem(image_id=1, text="a"),
                bge.CorpusItem(image_id=1, text="b"),
            ),
        )


def test_owned_export_is_pinned_to_the_published_repository() -> None:
    assert bge.EXPORT.repo == "listlessbird/bge-small-en-v1.5-onnx"
    assert bge.EXPORT.revision == "d46fcc3e67304e574e08e911ce7e50d71bb728cf"
    assert bge.EXPORT.model_sha256 == (
        "6fb40fbcdf3dcc7a3fed12d56ff2d1324f69d0b7fd6c5afe05f4530a6142fdf8"
    )


def test_remote_result_must_exactly_match_the_requested_batch() -> None:
    request = bge.EncodeBatch(
        document_content_sha256="4" * 64,
        export=_export(),
        items=(bge.CorpusItem(image_id=1, text="Titles: Cat"),),
    )
    result = bge.EncodedBatch(
        document_content_sha256=request.document_content_sha256,
        export=request.export,
        items=(bge.Vector(image_id=1, values=(1.0, *([0.0] * 383))),),
    )
    bge.validate_result(request, result)

    with pytest.raises(ValueError, match="image IDs"):
        bge.validate_result(request, result.model_copy(update={"items": ()}))
