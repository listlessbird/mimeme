from __future__ import annotations

from tests.support.storage import Memory

from mimeme import index, storage
from mimeme.index import documents
from mimeme.index.ops import validate
from mimeme.search.document import SearchDocument


def _manifest(version: str) -> index.Manifest:
    digest = storage.Checksum.of(b"data").value
    return index.Manifest(
        version=version,
        target_generation=4,
        model="test/embed",
        index_type="flat",
        encoder=index.Encoder(repo="encoder", revision="rev", variant="model.onnx"),
        dimension=2,
        image_count=1,
        files=[
            index.File(name=name, key=f"indexes/{version}/{name}", sha256=digest, length=4)
            for name in ("index.faiss", "mapping.json", "metadata.json")
        ],
        complete_key=f"indexes/{version}/complete.json",
    )


async def test_validation_requires_published_manifest_and_every_artifact() -> None:
    artifacts = Memory()
    manifest = _manifest("v2")
    for file in manifest.files:
        await artifacts.put_bytes(storage.Object(file.key), b"data", content_type="x")
    await artifacts.put_bytes(
        storage.Object(manifest.complete_key),
        manifest.model_dump_json().encode(),
        content_type="application/json",
    )

    assert await validate(artifacts, manifest) == manifest


async def test_validation_reads_and_hashes_v2_documents() -> None:
    artifacts = Memory()
    legacy = _manifest("v2-documents")
    document_file = await documents.publish(
        artifacts,
        version=legacy.version,
        documents=[SearchDocument(image_id=1, titles=("Alias",))],
    )
    manifest = index.Manifest.model_validate(
        {**legacy.model_dump(), "format_version": 2, "documents": document_file}
    )
    for file in manifest.files:
        await artifacts.put_bytes(storage.Object(file.key), b"data", content_type="x")
    await artifacts.put_bytes(
        storage.Object(manifest.complete_key),
        manifest.model_dump_json().encode(),
        content_type="application/json",
    )

    assert await validate(artifacts, manifest) == manifest
