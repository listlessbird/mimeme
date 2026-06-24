import { env } from "@/env";
import type { components, paths } from "@/lib/api/schema";
import { logInfo, serializeError } from "@/lib/observability";
import { queryOptions } from "@tanstack/react-query";
import { createServerFn } from "@tanstack/react-start";
import createClient from "openapi-fetch";
import { z } from "zod";

type Schemas = components["schemas"];

const errorBodySchema = z.object({
	detail: z.union([z.string(), z.array(z.object({ msg: z.string().optional() }))]).optional(),
});

export type Source = Schemas["SourceResponse"];
export type SourceListItem = Schemas["SourceListItemResponse"];
export type SourceListResponse = Schemas["SourceListResponse"];
export type SourceDetail = Schemas["SourceDetailResponse"];
export type SourceRun = Schemas["SourceRunResponse"];
export type SourceStats = Schemas["SourceStatsResponse"];
export type SourceItem = Schemas["SourceItemResponse"];
export type SourceItemListResponse = Schemas["SourceItemListResponse"];
export type RunItem = Schemas["RunItemResponse"];
export type RunItemListResponse = Schemas["RunItemListResponse"];
export type CreateSourceRequest = Schemas["CreateSourceRequest"];
export type UpdateSourceRequest = Schemas["UpdateSourceRequest"];
export type TriggerRunResponse = Schemas["TriggerRunResponse"];
export type MemeApiRawMetadata = Schemas["MemeApiRawMetadata"];
export type ProcessingStatus = Schemas["ProcessingStatus"];
export type SourceRunStatus = Schemas["SourceRunStatus"];
export type SourceRunTrigger = Schemas["SourceRunTrigger"];
export type DuplicateReason = Schemas["DuplicateReason"];

export class AdminApiError extends Error {
	readonly statusCode?: number;
	readonly detail?: string;

	constructor(
		message: string,
		options?: { statusCode?: number; detail?: string; cause?: unknown },
	) {
		super(message, { cause: options?.cause });
		this.name = "AdminApiError";
		this.statusCode = options?.statusCode;
		this.detail = options?.detail;
	}
}

function adminClient() {
	return createClient<paths>({
		baseUrl: env.API_BASE_URL,
		headers: { "X-API-Key": env.API_KEY_ADMIN ?? "" },
	});
}

function extractDetail(error: unknown): string | undefined {
	const parsed = errorBodySchema.safeParse(error);
	if (!parsed.success) return undefined;
	const detail = parsed.data.detail;
	if (typeof detail === "string") return detail;
	if (Array.isArray(detail)) {
		const messages = detail.map((entry) => entry.msg).filter(Boolean);
		if (messages.length) return messages.join("; ");
	}
	return undefined;
}

interface ClientResult<T> {
	data?: T;
	error?: unknown;
	response: Response;
}

async function callAdmin<T>(
	operation: string,
	request: (client: ReturnType<typeof adminClient>) => Promise<ClientResult<T>>,
): Promise<T> {
	const start = Date.now();
	const event: Record<string, unknown> = { operation };

	try {
		const { data, error, response } = await request(adminClient());
		event.status_code = response.status;

		if (error !== undefined || !response.ok) {
			const detail = extractDetail(error);
			event.outcome = "error";
			throw new AdminApiError(detail ?? `${operation} failed`, {
				statusCode: response.status,
				detail,
			});
		}

		if (data === undefined) {
			throw new AdminApiError(`${operation} returned no data`, {
				statusCode: response.status,
			});
		}
		event.outcome = "success";
		return data;
	} catch (error) {
		event.outcome ??= "error";
		event.error = serializeError(error);
		if (error instanceof AdminApiError) throw error;
		throw new AdminApiError(`${operation} request failed`, { cause: error });
	} finally {
		event.duration_ms = Date.now() - start;
		logInfo(`admin_api.${operation}`, event);
	}
}

export const listSources = createServerFn({ method: "GET" }).handler(() =>
	callAdmin<SourceListResponse>("list_sources", (c) => c.GET("/sources")),
);

export const getSource = createServerFn({ method: "GET" })
	.inputValidator((input: { id: number }) => input)
	.handler(({ data }) =>
		callAdmin<SourceDetail>("get_source", (c) =>
			c.GET("/sources/{source_id}", { params: { path: { source_id: data.id } } }),
		),
	);

export const createSource = createServerFn({ method: "POST" })
	.inputValidator((input: CreateSourceRequest) => input)
	.handler(({ data }) =>
		callAdmin<Source>("create_source", (c) => c.POST("/sources", { body: data })),
	);

export const updateSource = createServerFn({ method: "POST" })
	.inputValidator((input: { id: number; body: UpdateSourceRequest }) => input)
	.handler(({ data }) =>
		callAdmin<Source>("update_source", (c) =>
			c.PATCH("/sources/{source_id}", {
				params: { path: { source_id: data.id } },
				body: data.body,
			}),
		),
	);

export const deleteSource = createServerFn({ method: "POST" })
	.inputValidator((input: { id: number }) => input)
	.handler(({ data }) =>
		callAdmin<void>("delete_source", (c) =>
			c.DELETE("/sources/{source_id}", { params: { path: { source_id: data.id } } }),
		),
	);

export const triggerRun = createServerFn({ method: "POST" })
	.inputValidator((input: { id: number }) => input)
	.handler(({ data }) =>
		callAdmin<TriggerRunResponse>("trigger_run", (c) =>
			c.POST("/sources/{source_id}/run", { params: { path: { source_id: data.id } } }),
		),
	);

export const listSourceItems = createServerFn({ method: "GET" })
	.inputValidator((input: { id: number; limit?: number; offset?: number }) => input)
	.handler(({ data }) =>
		callAdmin<SourceItemListResponse>("list_source_items", (c) =>
			c.GET("/sources/{source_id}/items", {
				params: {
					path: { source_id: data.id },
					query: { limit: data.limit, offset: data.offset },
				},
			}),
		),
	);

export const listRunItems = createServerFn({ method: "GET" })
	.inputValidator((input: { id: number; runId: number; limit?: number; offset?: number }) => input)
	.handler(({ data }) =>
		callAdmin<RunItemListResponse>("list_run_items", (c) =>
			c.GET("/sources/{source_id}/runs/{run_id}/items", {
				params: {
					path: { source_id: data.id, run_id: data.runId },
					query: { limit: data.limit, offset: data.offset },
				},
			}),
		),
	);

export const SOURCE_DETAIL_POLL_MS = 10_000;
export const ITEMS_PAGE_SIZE = 24;

export const adminQueryKeys = {
	sources: ["admin", "sources"] as const,
	source: (id: number) => ["admin", "source", id] as const,
	sourceItems: (id: number, offset: number) =>
		["admin", "source", id, "items", { offset }] as const,
	runItems: (id: number, runId: number, offset: number) =>
		["admin", "source", id, "runs", runId, "items", { offset }] as const,
};

export const sourcesQueryOptions = () =>
	queryOptions({
		queryKey: adminQueryKeys.sources,
		queryFn: () => listSources(),
	});

export const sourceQueryOptions = (id: number) =>
	queryOptions({
		queryKey: adminQueryKeys.source(id),
		queryFn: () => getSource({ data: { id } }),
		// Light polling so manual and scheduled runs surface, and their counts
		// settle, without a manual refresh.
		refetchInterval: SOURCE_DETAIL_POLL_MS,
	});

export const sourceItemsQueryOptions = (id: number, offset: number) =>
	queryOptions({
		queryKey: adminQueryKeys.sourceItems(id, offset),
		queryFn: () => listSourceItems({ data: { id, limit: ITEMS_PAGE_SIZE, offset } }),
	});

export const runItemsQueryOptions = (id: number, runId: number, offset: number) =>
	queryOptions({
		queryKey: adminQueryKeys.runItems(id, runId, offset),
		queryFn: () => listRunItems({ data: { id, runId, limit: ITEMS_PAGE_SIZE, offset } }),
	});
