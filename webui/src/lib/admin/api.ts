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

const imageIngestResponseSchema = z.object({
	job_id: z.string(),
	queued: z.number(),
	duplicates: z.number(),
	message: z.string(),
});

export type Source = Schemas["SourceResponse"];
export type SourceListItem = Schemas["SourceListItemResponse"];
export type SourceListResponse = Schemas["SourceListResponse"];
export type SourceDetail = Schemas["SourceDetailResponse"];
export type SourceRun = Schemas["SourceRunResponse"];
export type SourceStats = Schemas["SourceStatsResponse"];
export type SourceItem = Schemas["SourceItemResponse"];
export type SourceItemListResponse = Schemas["SourceItemListResponse"];
export type SourceItemIngestState = Schemas["SourceItemIngestState"];
export type RunItem = Schemas["RunItemResponse"];
export type RunItemListResponse = Schemas["RunItemListResponse"];
export type CreateSourceRequest = Schemas["CreateSourceRequest"];
export type UpdateSourceRequest = Schemas["UpdateSourceRequest"];
export type TriggerRunResponse = Schemas["TriggerRunResponse"];
export type RetryResponse = Schemas["RetryResponse"];
export type MemeApiRawMetadata = Schemas["MemeApiRawMetadata"];
export type ProcessingStatus = Schemas["ProcessingStatus"];
export type SourceRunStatus = Schemas["SourceRunStatus"];
export type SourceRunTrigger = Schemas["SourceRunTrigger"];
export type DuplicateReason = Schemas["DuplicateReason"];
export type Image = Schemas["ImageResponse"];
export type ImageListResponse = Schemas["ImageListResponse"];
export type ImageIngestResponse = Schemas["ImageIngestResponse"];
export type ImageStatus = Schemas["ImageStatus"];
export type JobStatus = Schemas["JobStatus"];
export type JobType = Schemas["JobType"];
export type JobResult =
	| Schemas["IngestJobResult"]
	| Schemas["RebuildJobResult"]
	| Schemas["RawJobResult"];
export type Job = Omit<Schemas["JobResponse"], "result"> & {
	result?: JobResult | null;
};
export type IngestionRow = Schemas["IngestionRowResponse"];
export type IngestionListResponse = Schemas["IngestionListResponse"];
export type IngestionDetail = Schemas["IngestionDetailResponse"];
export type IngestionView = Schemas["IngestionView"];
export type IngestStage = Schemas["IngestStage"];
export type IngestOutcome = Schemas["IngestOutcome"];
export type IngestionLogs = Schemas["IngestionLogsResponse"];
export type IngestionLogEntry = Schemas["IngestionLogEntryResponse"];
export type ImageIngestInput = Schemas["RemoteImageUrlInput"] | Schemas["StagedUploadInput"];

export function describeImageIngestInput(input: ImageIngestInput): string {
	switch (input.kind) {
		case "remote_image_url":
			return input.url;
		case "staged_upload": {
			const filename = input.artifact_key.split("/").at(-1);
			return filename ? `Uploaded file: ${filename}` : "Uploaded file";
		}
	}
}

export const IMAGE_STATUSES: ImageStatus[] = [
	"pending",
	"downloading",
	"scanning",
	"annotating",
	"embedding",
	"done",
	"failed",
];

export function isImageStatus(value: string): value is ImageStatus {
	return IMAGE_STATUSES.some((status) => status === value);
}

export type ImageSort = "newest" | "oldest";

export const IMAGE_SORTS: ImageSort[] = ["newest", "oldest"];

export function isImageSort(value: string): value is ImageSort {
	return IMAGE_SORTS.some((sort) => sort === value);
}

export interface ImageListParams {
	limit?: number;
	offset?: number;
	status?: ImageStatus | null;
	dataset?: string | null;
	sort?: ImageSort;
}

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

export function adminErrorMessage(error: unknown, fallback: string): string {
	return error instanceof AdminApiError && error.detail ? error.detail : fallback;
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

export const retrySource = createServerFn({ method: "POST" })
	.inputValidator((input: { id: number }) => input)
	.handler(({ data }) =>
		callAdmin<RetryResponse>("retry_source", (c) =>
			c.POST("/sources/{source_id}/retry", { params: { path: { source_id: data.id } } }),
		),
	);

export const retrySourceRun = createServerFn({ method: "POST" })
	.inputValidator((input: { id: number; runId: number }) => input)
	.handler(({ data }) =>
		callAdmin<RetryResponse>("retry_source_run", (c) =>
			c.POST("/sources/{source_id}/runs/{run_id}/retry", {
				params: { path: { source_id: data.id, run_id: data.runId } },
			}),
		),
	);

export const retrySourceItem = createServerFn({ method: "POST" })
	.inputValidator((input: { id: number; itemId: number }) => input)
	.handler(({ data }) =>
		callAdmin<RetryResponse>("retry_source_item", (c) =>
			c.POST("/sources/{source_id}/items/{item_id}/retry", {
				params: { path: { source_id: data.id, item_id: data.itemId } },
			}),
		),
	);

export const listSourceItems = createServerFn({ method: "GET" })
	.inputValidator(
		(input: { id: number; limit?: number; offset?: number; status?: SourceItemIngestState }) =>
			input,
	)
	.handler(({ data }) =>
		callAdmin<SourceItemListResponse>("list_source_items", (c) =>
			c.GET("/sources/{source_id}/items", {
				params: {
					path: { source_id: data.id },
					query: { limit: data.limit, offset: data.offset, status: data.status },
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

export const listImages = createServerFn({ method: "GET" })
	.inputValidator((input: ImageListParams) => input)
	.handler(({ data }) =>
		callAdmin<ImageListResponse>("list_images", (c) =>
			c.GET("/images", {
				params: {
					query: {
						limit: data.limit,
						offset: data.offset,
						status: data.status ?? undefined,
						dataset: data.dataset || undefined,
						sort: data.sort,
					},
				},
			}),
		),
	);

export const ingestImageUrls = createServerFn({ method: "POST" })
	.inputValidator((input: { urls: string[]; dataset?: string | null; tags?: string[] }) => input)
	.handler(({ data }) =>
		callAdmin<ImageIngestResponse>("ingest_images", (c) =>
			c.POST("/images", {
				body: { urls: data.urls, dataset: data.dataset || undefined, tags: data.tags ?? [] },
			}),
		),
	);

export const uploadImageFile = createServerFn({ method: "POST" })
	.inputValidator((data: FormData) => data)
	.handler(async ({ data }) => {
		const start = Date.now();
		const event: Record<string, unknown> = { operation: "upload_image" };
		try {
			const res = await fetch(new URL("/images/upload", env.API_BASE_URL), {
				method: "POST",
				headers: { "X-API-Key": env.API_KEY_ADMIN ?? "" },
				body: data,
			});
			event.status_code = res.status;
			if (!res.ok) {
				event.outcome = "error";
				throw new AdminApiError(`upload_image failed`, { statusCode: res.status });
			}
			event.outcome = "success";
			return imageIngestResponseSchema.parse(await res.json());
		} catch (error) {
			event.outcome ??= "error";
			event.error = serializeError(error);
			if (error instanceof AdminApiError) throw error;
			throw new AdminApiError("upload_image request failed", { cause: error });
		} finally {
			event.duration_ms = Date.now() - start;
			logInfo("admin_api.upload_image", event);
		}
	});

export interface IngestionListParams {
	view: IngestionView;
	stage?: IngestStage | null;
	trigger?: SourceRunTrigger | null;
	source_id?: number | null;
	dataset?: string | null;
	outcome?: IngestOutcome | null;
	created_from?: string | null;
	created_to?: string | null;
	limit?: number;
	offset?: number;
}

export const listIngestion = createServerFn({ method: "GET" })
	.inputValidator((input: IngestionListParams) => input)
	.handler(({ data }) =>
		callAdmin<IngestionListResponse>("list_ingestion", (c) =>
			c.GET("/ingestion", {
				params: {
					query: {
						view: data.view,
						stage: data.stage ?? undefined,
						trigger: data.trigger ?? undefined,
						source_id: data.source_id ?? undefined,
						dataset: data.dataset || undefined,
						outcome: data.outcome ?? undefined,
						created_from: data.created_from ?? undefined,
						created_to: data.created_to ?? undefined,
						limit: data.limit,
						offset: data.offset,
					},
				},
			}),
		),
	);

export const getIngestionAttempt = createServerFn({ method: "GET" })
	.inputValidator((input: { id: number }) => input)
	.handler(({ data }) =>
		callAdmin<IngestionDetail>("get_ingestion_attempt", (c) =>
			c.GET("/ingestion/{ingest_url_id}", {
				params: { path: { ingest_url_id: data.id } },
			}),
		),
	);

export const getIngestionLogs = createServerFn({ method: "GET" })
	.inputValidator((input: { id: number; limit?: number }) => input)
	.handler(({ data }) =>
		callAdmin<IngestionLogs>("get_ingestion_logs", (c) =>
			c.GET("/ingestion/{ingest_url_id}/logs", {
				params: {
					path: { ingest_url_id: data.id },
					query: { limit: data.limit },
				},
			}),
		),
	);

export const getImage = createServerFn({ method: "GET" })
	.inputValidator((input: { id: number }) => input)
	.handler(({ data }) =>
		callAdmin<Image>("get_image", (c) =>
			c.GET("/images/{image_id}", { params: { path: { image_id: data.id } } }),
		),
	);

export const getJob = createServerFn({ method: "GET" })
	.inputValidator((input: { id: string }) => input)
	.handler(async ({ data }) => {
		const job = await callAdmin<Schemas["JobResponse"]>("get_job", (c) =>
			c.GET("/jobs/{job_id}", { params: { path: { job_id: data.id } } }),
		);
		// SAFETY: narrow the generated open `JobResult` union (which includes a
		// `{ [key: string]: JsonValue }` catch-all that cannot round-trip through
		// the server-fn boundary) to the three concrete result shapes. `result`
		// is display-only (stringified), so narrowing is sound here.
		return job as Job;
	});

export const SOURCE_DETAIL_POLL_MS = 10_000;
export const ITEMS_PAGE_SIZE = 24;
export const IMAGES_PAGE_SIZE = 24;
export const JOB_POLL_MS = 2_000;
export const INGESTION_POLL_MS = 2_000;

export const adminQueryKeys = {
	sources: ["admin", "sources"] as const,
	source: (id: number) => ["admin", "source", id] as const,
	sourceItems: (id: number, offset: number, status?: SourceItemIngestState) =>
		["admin", "source", id, "items", { offset, status }] as const,
	sourceItemsAll: (id: number) => ["admin", "source", id, "items"] as const,
	runItems: (id: number, runId: number, offset: number) =>
		["admin", "source", id, "runs", runId, "items", { offset }] as const,
	runItemsAll: (id: number, runId: number) =>
		["admin", "source", id, "runs", runId, "items"] as const,
	images: (params: ImageListParams) => ["admin", "images", params] as const,
	image: (id: number) => ["admin", "image", id] as const,
	job: (id: string) => ["admin", "job", id] as const,
	ingestion: (params: IngestionListParams) => ["admin", "ingestion", params] as const,
	ingestionAttempt: (id: number) => ["admin", "ingestion", "attempt", id] as const,
	ingestionLogs: (id: number) => ["admin", "ingestion", "attempt", id, "logs"] as const,
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

export const sourceItemsQueryOptions = (
	id: number,
	offset: number,
	status?: SourceItemIngestState,
) =>
	queryOptions({
		queryKey: adminQueryKeys.sourceItems(id, offset, status),
		queryFn: () => listSourceItems({ data: { id, limit: ITEMS_PAGE_SIZE, offset, status } }),
	});

export const runItemsQueryOptions = (id: number, runId: number, offset: number) =>
	queryOptions({
		queryKey: adminQueryKeys.runItems(id, runId, offset),
		queryFn: () => listRunItems({ data: { id, runId, limit: ITEMS_PAGE_SIZE, offset } }),
	});

export const imagesQueryOptions = (params: ImageListParams) =>
	queryOptions({
		queryKey: adminQueryKeys.images(params),
		queryFn: () => listImages({ data: params }),
	});

export const imageQueryOptions = (id: number) =>
	queryOptions({
		queryKey: adminQueryKeys.image(id),
		queryFn: () => getImage({ data: { id } }),
	});

export const jobQueryOptions = (id: string) =>
	queryOptions({
		queryKey: adminQueryKeys.job(id),
		queryFn: () => getJob({ data: { id } }),
	});

export const ingestionQueryOptions = (params: IngestionListParams, { poll }: { poll: boolean }) =>
	queryOptions({
		queryKey: adminQueryKeys.ingestion(params),
		queryFn: () => listIngestion({ data: params }),
		refetchInterval: poll ? INGESTION_POLL_MS : false,
	});

export const ingestionAttemptQueryOptions = (id: number) =>
	queryOptions({
		queryKey: adminQueryKeys.ingestionAttempt(id),
		queryFn: () => getIngestionAttempt({ data: { id } }),
	});

export const ingestionLogsQueryOptions = (id: number, { poll }: { poll: boolean }) =>
	queryOptions({
		queryKey: adminQueryKeys.ingestionLogs(id),
		queryFn: () => getIngestionLogs({ data: { id } }),
		refetchInterval: poll ? INGESTION_POLL_MS : false,
	});
