import type { GifAnnotationItem } from "@/lib/admin/gif-annotation";
import { buildEvalExport } from "@/lib/admin/gif-annotation-export";
import { describe, expect, it } from "vitest";

const decisions = {
	visibleText: "accepted",
	visualDescription: "edited",
	sequenceDescription: "accepted",
	visualQueries: "accepted",
	captionQueries: "rejected",
	naturalQueries: "edited",
} as const;

function item(status: GifAnnotationItem["status"]): GifAnnotationItem {
	return {
		sha256: "a".repeat(64),
		position: 1,
		split: "tune",
		width: 320,
		height: 240,
		nFrames: 12,
		durationMs: 1200,
		nBytes: 1234,
		gifUrl: "https://example.com/test.gif",
		contactSheetUrl: "https://example.com/test.jpg",
		suggestion: null,
		suggestionModel: null,
		annotation: {
			visibleText: ["HELLO"],
			visualDescription: "person waves",
			sequenceDescription: "hand moves side to side",
			visualQueries: ["person waving"],
			captionQueries: [],
			naturalQueries: ["hello reaction gif"],
			notes: "",
			decisions,
		},
		status,
		revision: 1,
		updatedAt: "2026-07-10T00:00:00.000Z",
	};
}

describe("buildEvalExport", () => {
	it("exports completed query pairs with their type and provenance", () => {
		const result = buildEvalExport([item("complete"), item("draft"), item("skipped")]);

		expect(result.n_gifs).toBe(1);
		expect(result.n_queries).toBe(2);
		expect(result.pairs[0]?.queries).toEqual([
			{ text: "person waving", type: "visual" },
			{ text: "hello reaction gif", type: "natural" },
		]);
		expect(result.pairs[0]?.provenance).toEqual(decisions);
	});
});
