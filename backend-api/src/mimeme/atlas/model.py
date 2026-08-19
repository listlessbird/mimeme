from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TemplateAtlasRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    neighbors: int = Field(default=20, ge=5, le=100)
    similarity_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    min_cluster_size: int = Field(default=3, ge=2, le=100)


class TemplateAtlasImage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int
    url: str | None = None
    width: int | None = None
    height: int | None = None
    dataset: str | None = None
    title: str | None = None
    source: str | None = None
    similarity_to_medoid: float | None = None


class TemplateAnchor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    source: str | None = None
    source_item_id: int | None = None
    image_count: int = Field(ge=1)


class TemplateCluster(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    label: str
    size: int = Field(ge=1)
    medoid: TemplateAtlasImage
    anchors: list[TemplateAnchor] = Field(default_factory=list)
    samples: list[TemplateAtlasImage] = Field(default_factory=list)


class TemplateAtlas(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    generated_at: datetime
    model: str
    neighbors: int
    similarity_threshold: float
    min_cluster_size: int
    embedding_count: int
    clustered_image_count: int
    noise_image_count: int
    cluster_count: int
    graph_edge_count: int
    anchor_count: int
    clusters: list[TemplateCluster] = Field(default_factory=list)
