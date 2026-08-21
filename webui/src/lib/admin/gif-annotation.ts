import { adminGuard } from "@/lib/admin/guard";
import { queryOptions } from "@tanstack/react-query";
import { createServerFn } from "@tanstack/react-start";
import { env } from "cloudflare:workers";
import { z } from "zod";

export const GIF_ANNOTATION_FIELDS = [
	"visibleText",
	"visualDescription",
	"sequenceDescription",
	"visualQueries",
	"captionQueries",
	"naturalQueries",
] as const;

export type GifAnnotationField = (typeof GIF_ANNOTATION_FIELDS)[number];
export const suggestionDecisionSchema = z.enum(["pending", "accepted", "edited", "rejected"]);
export const gifAnnotationStatusSchema = z.enum(["draft", "complete", "skipped"]);

export type SuggestionDecision = z.infer<typeof suggestionDecisionSchema>;
export type GifAnnotationStatus = z.infer<typeof gifAnnotationStatusSchema>;

export interface GifSuggestion {
	visible_text: string[];
	visual_description: string;
	sequence_description: string;
	suggested_visual_queries: string[];
	suggested_caption_queries: string[];
	suggested_natural_queries: string[];
	uncertainty: string;
	supporting_frame_numbers: number[];
}

const annotationTextSchema = z.string().trim().max(2_000, "Keep this under 2,000 characters");
const annotationListSchema = z
	.array(annotationTextSchema)
	.max(12, "Use at most 12 entries")
	.transform((values) => values.filter(Boolean));

export const gifAnnotationSchema = z.object({
	visibleText: annotationListSchema,
	visualDescription: annotationTextSchema,
	sequenceDescription: annotationTextSchema,
	visualQueries: annotationListSchema,
	captionQueries: annotationListSchema,
	naturalQueries: annotationListSchema,
	notes: annotationTextSchema,
	decisions: z.object({
		visibleText: suggestionDecisionSchema,
		visualDescription: suggestionDecisionSchema,
		sequenceDescription: suggestionDecisionSchema,
		visualQueries: suggestionDecisionSchema,
		captionQueries: suggestionDecisionSchema,
		naturalQueries: suggestionDecisionSchema,
	}),
});

export type GifAnnotationDocument = z.output<typeof gifAnnotationSchema>;
export type GifAnnotationFormInput = z.input<typeof gifAnnotationSchema>;

export interface GifAnnotationItem {
	sha256: string;
	position: number;
	split: "tune" | "holdout";
	width: number;
	height: number;
	nFrames: number;
	durationMs: number;
	nBytes: number;
	gifUrl: string;
	contactSheetUrl: string;
	suggestion: GifSuggestion | null;
	suggestionModel: string | null;
	annotation: GifAnnotationDocument;
	status: GifAnnotationStatus;
	revision: number;
	updatedAt: string | null;
}

export interface GifAnnotationListResponse {
	items: GifAnnotationItem[];
	completed: number;
	skipped: number;
	total: number;
}

interface DatasetItem {
	sha256: string;
	position: number;
	asset_key: string;
	contact_sheet_asset_key: string;
	split: "tune" | "holdout";
	width: number;
	height: number;
	n_frames: number;
	duration_ms: number;
	n_bytes: number;
}

interface Dataset {
	version: number;
	n_gifs: number;
	items: DatasetItem[];
}

interface Suggestions {
	version: number;
	model: string;
	items: Record<string, GifSuggestion>;
}

interface AnnotationRow {
	sha256: string;
	annotation_json: string;
	status: GifAnnotationStatus;
	revision: number;
	updated_at: string;
}

interface SavedRow {
	revision: number;
	updated_at: string;
}

const DATASET_KEY = "gif-annotation/v1/dataset.json";
const SUGGESTIONS_KEY = "gif-annotation/v1/suggestions.json";
function emptyDecisions(): Record<GifAnnotationField, SuggestionDecision> {
	return Object.fromEntries(GIF_ANNOTATION_FIELDS.map((field) => [field, "pending"])) as Record<
		GifAnnotationField,
		SuggestionDecision
	>;
}

export function emptyAnnotation(): GifAnnotationDocument {
	return {
		visibleText: [],
		visualDescription: "",
		sequenceDescription: "",
		visualQueries: [],
		captionQueries: [],
		naturalQueries: [],
		notes: "",
		decisions: emptyDecisions(),
	};
}

async function readJsonObject<T>(key: string): Promise<T> {
	const object = await env.GIF_ANNOTATION_BUCKET.get(key);
	if (!object) throw new Error(`Missing GIF annotation object: ${key}`);
	return object.json<T>();
}

async function loadCatalog(): Promise<{ dataset: Dataset; suggestions: Suggestions | null }> {
	const [dataset, suggestionsObject] = await Promise.all([
		readJsonObject<Dataset>(DATASET_KEY),
		env.GIF_ANNOTATION_BUCKET.get(SUGGESTIONS_KEY),
	]);
	const suggestions = suggestionsObject ? await suggestionsObject.json<Suggestions>() : null;
	return { dataset, suggestions };
}

export const listGifAnnotations = createServerFn({ method: "GET" })
	.middleware([adminGuard])
	.handler(async () => {
		const [{ dataset, suggestions }, rowsResult] = await Promise.all([
			loadCatalog(),
			env.GIF_ANNOTATION_DB.prepare(
				"SELECT sha256, annotation_json, status, revision, updated_at FROM gif_annotations",
			).all<AnnotationRow>(),
		]);
		const rows = new Map(rowsResult.results.map((row) => [row.sha256, row]));
		const publicBaseUrl = env.GIF_ANNOTATION_PUBLIC_BASE_URL.replace(/\/$/, "");
		const items = dataset.items.map<GifAnnotationItem>((item) => {
			const row = rows.get(item.sha256);
			let annotation = emptyAnnotation();
			if (row) {
				try {
					const savedAnnotation = gifAnnotationSchema.safeParse(JSON.parse(row.annotation_json));
					annotation = savedAnnotation.success ? savedAnnotation.data : emptyAnnotation();
				} catch {
					annotation = emptyAnnotation();
				}
			}
			return {
				sha256: item.sha256,
				position: item.position,
				split: item.split,
				width: item.width,
				height: item.height,
				nFrames: item.n_frames,
				durationMs: item.duration_ms,
				nBytes: item.n_bytes,
				gifUrl: `${publicBaseUrl}/${item.asset_key}`,
				contactSheetUrl: `${publicBaseUrl}/${item.contact_sheet_asset_key}`,
				suggestion: suggestions?.items[item.sha256] ?? null,
				suggestionModel: suggestions?.model ?? null,
				annotation,
				status: row?.status ?? "draft",
				revision: row?.revision ?? 0,
				updatedAt: row?.updated_at ?? null,
			};
		});
		return {
			items,
			completed: items.filter((item) => item.status === "complete").length,
			skipped: items.filter((item) => item.status === "skipped").length,
			total: items.length,
		} satisfies GifAnnotationListResponse;
	});

const saveGifAnnotationSchema = z.object({
	sha256: z.string().regex(/^[a-f0-9]{64}$/, "Invalid GIF sha256"),
	revision: z.number().int().nonnegative(),
	status: gifAnnotationStatusSchema,
	annotation: gifAnnotationSchema,
});

export const saveGifAnnotation = createServerFn({ method: "POST" })
	.middleware([adminGuard])
	.inputValidator(saveGifAnnotationSchema)
	.handler(async ({ data }) => {
		const { dataset } = await loadCatalog();
		if (!dataset.items.some((item) => item.sha256 === data.sha256)) {
			throw new Error("GIF is not part of the annotation dataset");
		}
		const annotationJson = JSON.stringify(data.annotation);
		let saved: SavedRow | null;
		if (data.revision === 0) {
			saved = await env.GIF_ANNOTATION_DB.prepare(
				`INSERT INTO gif_annotations (sha256, annotation_json, status)
				 VALUES (?1, ?2, ?3)
				 ON CONFLICT(sha256) DO NOTHING
				 RETURNING revision, updated_at`,
			)
				.bind(data.sha256, annotationJson, data.status)
				.first<SavedRow>();
		} else {
			saved = await env.GIF_ANNOTATION_DB.prepare(
				`UPDATE gif_annotations
				 SET annotation_json = ?1,
				     status = ?2,
				     revision = revision + 1,
				     updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
				 WHERE sha256 = ?3 AND revision = ?4
				 RETURNING revision, updated_at`,
			)
				.bind(annotationJson, data.status, data.sha256, data.revision)
				.first<SavedRow>();
		}
		if (!saved) throw new Error("Annotation changed in another tab; reload before saving again");
		return { revision: saved.revision, updatedAt: saved.updated_at };
	});

export const gifAnnotationsQueryOptions = () =>
	queryOptions({
		queryKey: ["admin", "gif-annotations"] as const,
		queryFn: () => listGifAnnotations(),
		staleTime: 30_000,
	});
