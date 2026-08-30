from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mimeme.search import recipe

type Intent = Literal["reaction", "situation", "visual", "template", "quote", "conceptual"]
type QuerySource = Literal["human", "production", "synthetic"]
type QueryStatus = Literal["active", "disabled"]
type RunMode = Literal["image", "hybrid"]
type RunStatus = Literal["queued", "running", "needs_judgments", "complete", "failed", "cancelled"]
type RunPhase = Literal["preparing", "searching", "calculating_metrics", "finalizing"]


class WorkflowInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str


class FailureInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    error: str


class QuerySpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int
    text: str


class PreparedRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    recipe_id: recipe.RecipeId
    recipe: recipe.Definition
    index_version: str
    queries: list[QuerySpec]


class RetrievalBatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    recipe_id: recipe.RecipeId
    recipe: recipe.Definition
    index_version: str
    queries: list[QuerySpec]


class WorkflowResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    status: RunStatus


class QueryView(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int
    text: str
    intent: Intent
    source: QuerySource
    status: QueryStatus
    candidate_count: int = Field(ge=0)
    judgment_count: int = Field(ge=0)
    relevant_count: int = Field(ge=0)
    created_at: datetime


class CandidateView(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    image_id: int
    url: str | None
    caption: str | None
    ocr_text: str | None
    width: int | None
    height: int | None
    grade: int | None = Field(default=None, ge=0, le=3)
    revision: int = Field(ge=0)


class JudgmentWorkspace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: QueryView
    candidates: list[CandidateView]
    previous_query_id: int | None
    next_query_id: int | None


class RunMetricsView(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query_count: int
    ndcg_at_10: float
    precision_at_5: float
    success_at_5: float
    mrr_at_10: float
    judged_at_10: float
    latency_p50_ms: float
    latency_p95_ms: float


class RunView(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    experiment_id: str | None
    recipe_id: recipe.RecipeId
    recipe: recipe.Definition
    mode: RunMode
    status: RunStatus
    phase: RunPhase | None
    progress_completed: int = Field(ge=0)
    progress_total: int = Field(ge=0)
    index_version: str | None
    release_id: str
    snapshot_id: str
    metrics: RunMetricsView | None
    missing_judgments: int = Field(ge=0)
    error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class ExperimentView(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    snapshot_id: str
    index_version: str | None
    recipes: tuple[recipe.Definition, ...]
    runs: list[RunView]


class Overview(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    queries: list[QueryView]
    recent_runs: list[RunView]
    baseline_run_id: str | None
    active_query_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    judgment_count: int = Field(ge=0)
    unjudged_count: int = Field(ge=0)
    recipes: tuple[recipe.Definition, ...]


class PoolResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: int
    candidate_count: int
    added_count: int
    index_version: str
    batch_id: str


class JudgmentSave(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: int
    image_id: int
    grade: int = Field(ge=0, le=3)
    revision: int = Field(ge=1)


class RankedImage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    image_id: int
    rank: int
    url: str | None
    grade: int | None


class QueryComparison(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: int
    text: str
    intent: Intent
    baseline_ndcg_at_10: float
    candidate_ndcg_at_10: float
    delta_ndcg_at_10: float
    baseline_results: list[RankedImage]
    candidate_results: list[RankedImage]


class Comparison(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    baseline: RunView
    candidate: RunView
    delta_ndcg_at_10: float
    delta_success_at_5: float
    delta_mrr_at_10: float
    delta_latency_p95_ms: float
    improved_queries: int
    unchanged_queries: int
    regressed_queries: int
    queries: list[QueryComparison]
