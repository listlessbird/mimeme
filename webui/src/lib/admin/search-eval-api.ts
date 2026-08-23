import { callAdmin } from "@/lib/admin/api";
import { adminGuard } from "@/lib/admin/guard";
import type { components } from "@/lib/api/schema";
import { queryOptions } from "@tanstack/react-query";
import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";

type Schemas = components["schemas"];

export type SearchEvalOverview = Schemas["Overview"];
export type SearchEvalQuery = Schemas["QueryView"];
export type SearchEvalJudgmentWorkspace = Schemas["JudgmentWorkspace"];
export type SearchEvalRun = Schemas["RunView"];
export type SearchEvalComparison = Schemas["Comparison"];
export type SearchEvalIntent = Schemas["Intent"];
export type SearchEvalQuerySource = Schemas["QuerySource"];
export type SearchEvalRunMode = Schemas["RunMode"];

/** Intents supported by the first Core Search query set. */
export const SEARCH_EVAL_INTENTS: ReadonlyArray<SearchEvalIntent> = [
	"reaction",
	"situation",
	"visual",
	"template",
	"quote",
	"conceptual",
];

const createQueryInput = z.object({
	text: z.string().trim().min(1).max(200),
	intent: z.enum(SEARCH_EVAL_INTENTS),
	source: z.enum(["human", "production", "synthetic"]),
});

const queryIdInput = z.object({ queryId: z.number().int().positive() });
const judgmentInput = z.object({
	queryId: z.number().int().positive(),
	imageId: z.number().int().positive(),
	grade: z.number().int().min(0).max(3),
	revision: z.number().int().nonnegative(),
});

/** Load the Core Search workspace and recent runs. */
export const getSearchEvalOverview = createServerFn({ method: "GET" })
	.middleware([adminGuard])
	.handler(() =>
		callAdmin<SearchEvalOverview>("search_eval_overview", (client) => client.GET("/search-evals")),
	);

/** Create a user-centered Core Search query. */
export const createSearchEvalQuery = createServerFn({ method: "POST" })
	.middleware([adminGuard])
	.inputValidator(createQueryInput)
	.handler(({ data }) =>
		callAdmin<SearchEvalQuery>("create_search_eval_query", (client) =>
			client.POST("/search-evals/queries", { body: data }),
		),
	);

/** Disable a query without deleting its historical run data. */
export const disableSearchEvalQuery = createServerFn({ method: "POST" })
	.middleware([adminGuard])
	.inputValidator(queryIdInput)
	.handler(({ data }) =>
		callAdmin<void>("disable_search_eval_query", (client) =>
			client.DELETE("/search-evals/queries/{query_id}", {
				params: { path: { query_id: data.queryId } },
			}),
		),
	);

/** Pool the current image and hybrid top twenty for one query. */
export const poolSearchEvalQuery = createServerFn({ method: "POST" })
	.middleware([adminGuard])
	.inputValidator(queryIdInput)
	.handler(({ data }) =>
		callAdmin<Schemas["PoolResult"]>("pool_search_eval_query", (client) =>
			client.POST("/search-evals/queries/{query_id}/pool", {
				params: { path: { query_id: data.queryId } },
			}),
		),
	);

/** Add an image that retrieval missed to a query's judgment pool. */
export const addSearchEvalCandidate = createServerFn({ method: "POST" })
	.middleware([adminGuard])
	.inputValidator(
		z.object({
			queryId: z.number().int().positive(),
			imageId: z.number().int().positive(),
		}),
	)
	.handler(({ data }) =>
		callAdmin<void>("add_search_eval_candidate", (client) =>
			client.POST("/search-evals/queries/{query_id}/candidates", {
				params: { path: { query_id: data.queryId } },
				body: { image_id: data.imageId },
			}),
		),
	);

/** Load one query and its pooled candidates for blind grading. */
export const getSearchEvalJudgments = createServerFn({ method: "GET" })
	.middleware([adminGuard])
	.inputValidator(z.object({ queryId: z.number().int().positive().optional() }))
	.handler(({ data }) =>
		callAdmin<SearchEvalJudgmentWorkspace>("search_eval_judgments", (client) =>
			client.GET("/search-evals/judgments", {
				params: { query: { query_id: data.queryId } },
			}),
		),
	);

/** Save a 0 to 3 relevance grade with optimistic concurrency. */
export const saveSearchEvalJudgment = createServerFn({ method: "POST" })
	.middleware([adminGuard])
	.inputValidator(judgmentInput)
	.handler(({ data }) =>
		callAdmin<Schemas["JudgmentSave"]>("save_search_eval_judgment", (client) =>
			client.PUT("/search-evals/queries/{query_id}/judgments/{image_id}", {
				params: { path: { query_id: data.queryId, image_id: data.imageId } },
				body: { grade: data.grade, revision: data.revision },
			}),
		),
	);

/** Return a candidate to the unjudged queue with optimistic concurrency. */
export const clearSearchEvalJudgment = createServerFn({ method: "POST" })
	.middleware([adminGuard])
	.inputValidator(
		z.object({
			queryId: z.number().int().positive(),
			imageId: z.number().int().positive(),
			revision: z.number().int().positive(),
		}),
	)
	.handler(({ data }) =>
		callAdmin<void>("clear_search_eval_judgment", (client) =>
			client.DELETE("/search-evals/queries/{query_id}/judgments/{image_id}", {
				params: {
					path: { query_id: data.queryId, image_id: data.imageId },
					query: { revision: data.revision },
				},
			}),
		),
	);

/** Execute the active query set against one current search mode. */
export const createSearchEvalRun = createServerFn({ method: "POST" })
	.middleware([adminGuard])
	.inputValidator(z.object({ mode: z.enum(["image", "hybrid"]) }))
	.handler(({ data }) =>
		callAdmin<SearchEvalRun>("create_search_eval_run", (client) =>
			client.POST("/search-evals/runs", { body: data }),
		),
	);

/** Recompute an incomplete run after its missing judgments are graded. */
export const finalizeSearchEvalRun = createServerFn({ method: "POST" })
	.middleware([adminGuard])
	.inputValidator(z.object({ runId: z.string().min(1).max(32) }))
	.handler(({ data }) =>
		callAdmin<SearchEvalRun>("finalize_search_eval_run", (client) =>
			client.POST("/search-evals/runs/{run_id}/finalize", {
				params: { path: { run_id: data.runId } },
			}),
		),
	);

/** Promote a complete run to the Core Search baseline. */
export const setSearchEvalBaseline = createServerFn({ method: "POST" })
	.middleware([adminGuard])
	.inputValidator(z.object({ runId: z.string().min(1).max(32) }))
	.handler(({ data }) =>
		callAdmin<SearchEvalRun>("set_search_eval_baseline", (client) =>
			client.PUT("/search-evals/runs/{run_id}/baseline", {
				params: { path: { run_id: data.runId } },
			}),
		),
	);

/** Compare two complete runs scored against the same frozen judgments. */
export const getSearchEvalComparison = createServerFn({ method: "GET" })
	.middleware([adminGuard])
	.inputValidator(
		z.object({
			baselineRunId: z.string().min(1).max(32),
			candidateRunId: z.string().min(1).max(32),
		}),
	)
	.handler(({ data }) =>
		callAdmin<SearchEvalComparison>("compare_search_eval_runs", (client) =>
			client.GET("/search-evals/compare", {
				params: {
					query: {
						baseline_run_id: data.baselineRunId,
						candidate_run_id: data.candidateRunId,
					},
				},
			}),
		),
	);

/** React Query options for the Core Search overview. */
export const searchEvalOverviewQueryOptions = () =>
	queryOptions({
		queryKey: ["admin", "search-evals"] as const,
		queryFn: () => getSearchEvalOverview(),
		staleTime: 5_000,
		refetchInterval: (query) =>
			query.state.data?.recent_runs.some(
				(run) => run.status === "queued" || run.status === "running",
			)
				? 1_500
				: false,
	});

/** React Query options for a blind judgment workspace. */
export const searchEvalJudgmentsQueryOptions = (queryId?: number) =>
	queryOptions({
		queryKey: ["admin", "search-evals", "judgments", queryId ?? null] as const,
		queryFn: () => getSearchEvalJudgments({ data: { queryId } }),
	});

/** React Query options for a baseline and candidate comparison. */
export const searchEvalComparisonQueryOptions = (baselineRunId: string, candidateRunId: string) =>
	queryOptions({
		queryKey: ["admin", "search-evals", "compare", baselineRunId, candidateRunId] as const,
		queryFn: () => getSearchEvalComparison({ data: { baselineRunId, candidateRunId } }),
		enabled: Boolean(baselineRunId && candidateRunId),
	});
