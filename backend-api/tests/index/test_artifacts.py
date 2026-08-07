from __future__ import annotations

from tests.support.storage import Memory

from mimeme import index, storage
from mimeme.index.ops import cleanup_incomplete, validate


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


async def test_partial_generation_cleanup_never_deletes_protected_or_complete_output() -> None:
    artifacts = Memory()
    partial = storage.Object("indexes/partial/index.faiss")
    await artifacts.put_bytes(partial, b"data", content_type="x")
    await cleanup_incomplete(artifacts, version="partial", protect=set())
    assert await artifacts.stat(partial) is None

    protected = storage.Object("indexes/active/index.faiss")
    await artifacts.put_bytes(protected, b"data", content_type="x")
    await cleanup_incomplete(artifacts, version="active", protect={"active"})
    assert await artifacts.stat(protected) is not None
