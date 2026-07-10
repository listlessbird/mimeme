import type { GifAnnotationItem } from "@/lib/admin/gif-annotation";

export function buildEvalExport(items: GifAnnotationItem[]) {
	const pairs = items
		.filter((item) => item.status === "complete")
		.map((item) => ({
			sha256: item.sha256,
			queries: [
				...item.annotation.visualQueries.map((text) => ({ text, type: "visual" as const })),
				...item.annotation.captionQueries.map((text) => ({ text, type: "caption" as const })),
				...item.annotation.naturalQueries.map((text) => ({ text, type: "natural" as const })),
			],
			split: item.split,
			provenance: item.annotation.decisions,
		}))
		.filter((pair) => pair.queries.length > 0);
	return {
		version: 1,
		holdout_fraction: 0.2,
		n_gifs: pairs.length,
		n_queries: pairs.reduce((total, pair) => total + pair.queries.length, 0),
		pairs,
	};
}
