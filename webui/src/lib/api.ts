import { env } from "@/env";
import { logInfo, serializeError } from "@/lib/observability";
import { infiniteQueryOptions } from "@tanstack/react-query";
import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";

export const SEARCH_RESULT_LIMIT = 40;

const searchResultSchema = z.object({
	id: z.number(),
	sha256: z.string(),
	score: z.number(),
	url: z.string(),
	caption: z.string(),
	ocr_text: z.string(),
	width: z.number(),
	height: z.number(),
});

const searchResponseSchema = z.object({
	query: z.string(),
	results: z.array(searchResultSchema),
	total: z.number(),
	limit: z.number(),
	offset: z.number(),
	has_more: z.boolean(),
	search_time_ms: z.number(),
});

export type SearchResult = z.infer<typeof searchResultSchema>;
export type SearchResponse = z.infer<typeof searchResponseSchema>;

export const searchMemesInfiniteQueryOptions = (q: string) =>
	infiniteQueryOptions({
		queryKey: ["search-memes", q, "hybrid", SEARCH_RESULT_LIMIT],
		queryFn: ({ pageParam }) =>
			searchMemes({
				data: {
					q,
					limit: SEARCH_RESULT_LIMIT,
					offset: pageParam,
				},
			}),
		initialPageParam: 0,
		getNextPageParam: (lastPage) =>
			lastPage.has_more ? lastPage.offset + lastPage.results.length : undefined,
	});

class SearchApiError extends Error {
	public readonly statusCode?: number;
	public readonly responseBody?: string;

	constructor(
		message: string,
		options?: { statusCode?: number; responseBody?: string; cause?: unknown },
	) {
		super(message, { cause: options?.cause });
		this.name = "SearchApiError";
		this.statusCode = options?.statusCode;
		this.responseBody = options?.responseBody;
	}
}

export const searchMemes = createServerFn({ method: "GET" })
	.inputValidator((input: { q: string; limit?: number; offset?: number }) => input)
	.handler(async ({ data }) => {
		const start = Date.now();
		const requestId = crypto.randomUUID();
		const endpointPath = "/search";
		const requestUrl = new URL(endpointPath, env.API_BASE_URL);
		requestUrl.searchParams.set("q", data.q);
		requestUrl.searchParams.set("mode", "hybrid");
		if (data.limit) requestUrl.searchParams.set("limit", String(data.limit));
		if (data.offset) requestUrl.searchParams.set("offset", String(data.offset));

		const wideEvent: Record<string, unknown> = {
			request_id: requestId,
			operation: "search_memes",
			method: "GET",
			endpoint: endpointPath,
			endpoint_url: requestUrl.toString(),
			query: data.q,
			query_length: data.q.length,
			mode: "hybrid",
			limit: data.limit,
			offset: data.offset,
		};

		try {
			const res = await fetch(requestUrl, {
				headers: {
					"X-API-Key": env.API_KEY_READONLY,
					"X-Request-ID": requestId,
				},
			});

			wideEvent.status_code = res.status;

			if (!res.ok) {
				const responseBody = await res.text();
				throw new SearchApiError("Search API returned a non-success status", {
					statusCode: res.status,
					responseBody,
				});
			}

			const payload = searchResponseSchema.parse(await res.json());
			wideEvent.outcome = "success";
			wideEvent.result_count = payload.results.length;
			wideEvent.search_time_ms = payload.search_time_ms;

			return payload;
		} catch (error) {
			wideEvent.outcome = "error";
			wideEvent.error = serializeError(error);

			if (error instanceof SearchApiError) {
				throw error;
			}

			throw new SearchApiError("Search API request failed", { cause: error });
		} finally {
			wideEvent.duration_ms = Date.now() - start;
			logInfo("search_memes.request", wideEvent);
		}
	});
