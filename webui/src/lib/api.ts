import { createServerFn } from "@tanstack/react-start";
import { env } from "@/env";

export interface SearchResult {
	id: number;
	sha256: string;
	score: number;
	url: string;
	caption: string;
	ocr_text: string;
	width: number;
	height: number;
}

export interface SearchResponse {
	query: string;
	results: SearchResult[];
	total: number;
	limit: number;
	offset: number;
	search_time_ms: number;
}

export const searchMemes = createServerFn({ method: "GET" })
	.inputValidator(
		(input: { q: string; limit?: number; offset?: number }) => input,
	)
	.handler(async ({ data }) => {
		const params = new URLSearchParams({ q: data.q });
		if (data.limit) params.set("limit", String(data.limit));
		if (data.offset) params.set("offset", String(data.offset));

		const res = await fetch(`${env.API_BASE_URL}/search?${params}`, {
			headers: {
				"X-API-Key": env.API_KEY_READONLY,
			},
		});

		if (!res.ok) {
			throw new Error(`API error: ${res.status} ${res.statusText}`);
		}

		return (await res.json()) as SearchResponse;
	});
