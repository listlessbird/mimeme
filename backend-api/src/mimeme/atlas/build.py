from __future__ import annotations

import asyncio
import io
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import or_, select

from mimeme import storage
from mimeme.atlas.model import (
    TemplateAnchor,
    TemplateAtlas,
    TemplateAtlasImage,
    TemplateAtlasRunRequest,
    TemplateCluster,
)
from mimeme.db import Db
from mimeme.db.schema import (
    Image,
    IngestionSource,
    IngestURL,
    Processing,
    ProcessingStatus,
    SourceItem,
)
from mimeme.media import Urls

_ATLAS_KEY = "atlases/siglip2-template-atlas.json"
_LABEL_SPACE_RE = re.compile(r"\s+")
_ANCHOR_KEYS = ("template", "template_name", "meme_template", "template_title")


class NoEmbeddings(Exception):
    """No completed image embeddings were available for an atlas run."""


@dataclass(frozen=True)
class _Anchor:
    key: str
    label: str
    source: str | None
    source_item_id: int
    strength: int


@dataclass(frozen=True)
class _Record:
    image_id: int
    url: str | None
    width: int | None
    height: int | None
    dataset: str | None
    source: str | None
    title: str | None
    anchor: _Anchor | None
    embedding_key: str


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self._rank[left_root] < self._rank[right_root]:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        if self._rank[left_root] == self._rank[right_root]:
            self._rank[left_root] += 1

    def find(self, value: int) -> int:
        root = value
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[value] != value:
            parent = self._parent[value]
            self._parent[value] = root
            value = parent
        return root


async def build_template_atlas(
    db: Db,
    artifacts: storage.Store,
    media_urls: Urls,
    *,
    model: str,
    options: TemplateAtlasRunRequest,
) -> TemplateAtlas:
    records = await _load_records(db, media_urls, model=model)
    if not records:
        raise NoEmbeddings()

    vectors = await _load_vectors(artifacts, records)
    if not vectors:
        raise NoEmbeddings()

    from typing import cast

    import faiss  # type: ignore[import-untyped]
    import numpy as np

    records = [record for record, _ in vectors]
    matrix = np.asarray([vector for _, vector in vectors], dtype=np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    faiss_module = cast(Any, faiss)
    index = faiss_module.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    neighbor_count = min(options.neighbors + 1, len(records))
    scores, neighbors = index.search(matrix, neighbor_count)
    disjoint_set = _DisjointSet(len(records))
    graph_edge_count = 0
    anchor_groups: dict[str, list[int]] = defaultdict(list)

    for position, record in enumerate(records):
        if record.anchor is not None:
            anchor_groups[record.anchor.key].append(position)
        for neighbor_position, score in zip(neighbors[position, 1:], scores[position, 1:]):
            if neighbor_position < 0 or float(score) < options.similarity_threshold:
                continue
            disjoint_set.union(position, int(neighbor_position))
            graph_edge_count += 1

    # A source item such as a KYM template gallery is a high-confidence anchor:
    # keep all of its ingested media together even when caption overlays drift.
    for positions in anchor_groups.values():
        for position in positions[1:]:
            disjoint_set.union(positions[0], position)

    components: dict[int, list[int]] = defaultdict(list)
    for position in range(len(records)):
        components[disjoint_set.find(position)].append(position)

    clusters: list[TemplateCluster] = []
    noise_image_count = 0
    cluster_number = 1
    for positions in sorted(components.values(), key=len, reverse=True):
        if len(positions) < options.min_cluster_size:
            noise_image_count += len(positions)
            continue
        cluster = _make_cluster(
            cluster_number,
            positions,
            records,
            matrix,
        )
        clusters.append(cluster)
        cluster_number += 1

    anchor_count = sum(1 for record in records if record.anchor is not None)
    atlas = TemplateAtlas(
        generated_at=datetime.now(UTC),
        model=model,
        neighbors=options.neighbors,
        similarity_threshold=options.similarity_threshold,
        min_cluster_size=options.min_cluster_size,
        embedding_count=len(records),
        clustered_image_count=sum(cluster.size for cluster in clusters),
        noise_image_count=noise_image_count,
        cluster_count=len(clusters),
        graph_edge_count=graph_edge_count,
        anchor_count=anchor_count,
        clusters=clusters,
    )
    await artifacts.put_bytes(
        storage.Object(_ATLAS_KEY),
        atlas.model_dump_json().encode(),
        content_type="application/json",
    )
    return atlas


async def load_template_atlas(artifacts: storage.Store) -> TemplateAtlas | None:
    raw = await artifacts.read_bytes(storage.Object(_ATLAS_KEY), max_bytes=5 * 1024 * 1024)
    return TemplateAtlas.model_validate_json(raw)


def atlas_key() -> str:
    return _ATLAS_KEY


async def _load_records(db: Db, media_urls: Urls, *, model: str) -> list[_Record]:
    async with db.read_session() as session:
        rows = (
            await session.execute(
                select(Image, Processing)
                .join(Processing, Processing.image_id == Image.id)
                .where(
                    Processing.embed_status == ProcessingStatus.DONE,
                    Processing.embed_s3_key.is_not(None),
                    Processing.embed_model == model,
                )
                .order_by(Image.id.asc())
            )
        ).all()
        if not rows:
            return []

        image_ids = [image.id for image, _ in rows]
        provenance_rows = (
            await session.execute(
                select(
                    IngestURL.image_id,
                    IngestURL.duplicate_of_image_id,
                    SourceItem.id,
                    SourceItem.title,
                    SourceItem.known_facts,
                    SourceItem.raw_metadata,
                    IngestionSource.name,
                    IngestionSource.adapter_key,
                )
                .outerjoin(SourceItem, SourceItem.id == IngestURL.source_item_id)
                .outerjoin(IngestionSource, IngestionSource.id == IngestURL.source_id)
                .where(
                    or_(
                        IngestURL.image_id.in_(image_ids),
                        IngestURL.duplicate_of_image_id.in_(image_ids),
                    )
                )
                .order_by(IngestURL.id.asc())
            )
        ).all()

    metadata_by_image: dict[int, _Anchor] = {}
    for (
        image_id,
        duplicate_of_image_id,
        source_item_id,
        title,
        known_facts,
        raw_metadata,
        source_name,
        adapter_key,
    ) in provenance_rows:
        target_id = image_id or duplicate_of_image_id
        if target_id is None or source_item_id is None:
            continue
        anchor = _make_anchor(
            source_item_id=source_item_id,
            title=title,
            known_facts=known_facts,
            raw_metadata=raw_metadata,
            source_name=source_name,
            adapter_key=adapter_key,
        )
        if anchor is not None and target_id not in metadata_by_image:
            metadata_by_image[target_id] = anchor

    records: list[_Record] = []
    for image, processing in rows:
        if processing.embed_s3_key is None:
            continue
        anchor = metadata_by_image.get(image.id)
        records.append(
            _Record(
                image_id=image.id,
                url=media_urls.resolve(image.s3_key) if image.s3_key else None,
                width=image.width,
                height=image.height,
                dataset=image.dataset,
                source=anchor.source if anchor else None,
                title=anchor.label if anchor else None,
                anchor=anchor,
                embedding_key=processing.embed_s3_key,
            )
        )
    return records


async def _load_vectors(
    artifacts: storage.Store, records: list[_Record]
) -> list[tuple[_Record, Any]]:
    import numpy as np

    semaphore = asyncio.Semaphore(16)

    async def load(record: _Record) -> tuple[_Record, Any] | None:
        try:
            async with semaphore:
                raw = await artifacts.read_bytes(storage.Object(record.embedding_key), max_bytes=16 * 1024 * 1024)
            vector = np.load(io.BytesIO(raw), allow_pickle=False)
            if vector.ndim != 1 or not np.issubdtype(vector.dtype, np.floating):
                return None
            if not np.all(np.isfinite(vector)) or float(np.linalg.norm(vector)) == 0:
                return None
            return record, vector.astype(np.float32, copy=False)
        except (storage.Error, ValueError, OSError):
            return None

    loaded = await asyncio.gather(*(load(record) for record in records))
    return [item for item in loaded if item is not None]


def _make_anchor(
    *,
    source_item_id: int,
    title: str | None,
    known_facts: dict[str, Any] | None,
    raw_metadata: dict[str, Any] | None,
    source_name: str | None,
    adapter_key: str | None,
) -> _Anchor | None:
    facts = known_facts if isinstance(known_facts, dict) else {}
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    explicit = next(
        (
            value
            for key in _ANCHOR_KEYS
            for value in [metadata.get(key), facts.get(key)]
            if isinstance(value, str) and value.strip()
        ),
        None,
    )
    label = explicit or (title.strip() if isinstance(title, str) and title.strip() else None)
    if label is None:
        return None

    is_kym = adapter_key == "kym"
    is_explicit = explicit is not None
    if not is_kym and not is_explicit:
        return None
    clean_label = _clean_label(label)
    return _Anchor(
        key=f"{adapter_key or 'source'}:{source_item_id}",
        label=clean_label,
        source=source_name,
        source_item_id=source_item_id,
        strength=3 if is_kym else 2,
    )


def _make_cluster(
    number: int,
    positions: list[int],
    records: list[_Record],
    matrix: Any,
) -> TemplateCluster:
    import numpy as np

    centroid = matrix[positions].mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    medoid_position = max(positions, key=lambda position: float(matrix[position] @ centroid))

    anchors_by_key: dict[str, list[_Anchor]] = defaultdict(list)
    for position in positions:
        anchor = records[position].anchor
        if anchor is not None:
            anchors_by_key[anchor.key].append(anchor)
    anchors = [
        TemplateAnchor(
            label=items[0].label,
            source=items[0].source,
            source_item_id=items[0].source_item_id,
            image_count=len(items),
        )
        for items in sorted(
            anchors_by_key.values(),
            key=lambda items: (items[0].strength, len(items), items[0].label),
            reverse=True,
        )[:5]
    ]
    label = anchors[0].label if anchors else f"cluster {number:03d}"
    medoid = _image_view(records[medoid_position], similarity=1.0)

    candidate_positions = sorted(
        positions,
        key=lambda position: float(matrix[position] @ matrix[medoid_position]),
        reverse=True,
    )
    samples = [
        _image_view(
            records[position],
            similarity=float(matrix[position] @ matrix[medoid_position]),
        )
        for position in candidate_positions[:12]
    ]
    return TemplateCluster(
        id=f"cluster-{number:03d}",
        label=label,
        size=len(positions),
        medoid=medoid,
        anchors=anchors,
        samples=samples,
    )


def _image_view(record: _Record, *, similarity: float) -> TemplateAtlasImage:
    return TemplateAtlasImage(
        id=record.image_id,
        url=record.url,
        width=record.width,
        height=record.height,
        dataset=record.dataset,
        title=record.title,
        source=record.source,
        similarity_to_medoid=round(similarity, 4),
    )


def _clean_label(value: str) -> str:
    return _LABEL_SPACE_RE.sub(" ", value.strip()).strip()
