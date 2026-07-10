import { checkAdminAccess } from "@/lib/admin/guard";
import { queryOptions } from "@tanstack/react-query";
import { createServerFn } from "@tanstack/react-start";
import { env } from "cloudflare:workers";

export const GIF_ANNOTATION_FIELDS = [
	"visibleText",
	"visualDescription",
	"sequenceDescription",
	"visualQueries",
	"captionQueries",
	"naturalQueries",
] as const;

export type GifAnnotationField = (typeof GIF_ANNOTATION_FIELDS)[number];
export type SuggestionDecision = "pending" | "accepted" | "edited" | "rejected";
export type GifAnnotationStatus = "draft" | "complete" | "skipped";

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

export interface GifAnnotationDocument {
	visibleText: string[];
	visualDescription: string;
	sequenceDescription: string;
	visualQueries: string[];
	captionQueries: string[];
	naturalQueries: string[];
	notes: string;
	decisions: Record<GifAnnotationField, SuggestionDecision>;
}

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
const MAX_TEXT_LENGTH = 2_000;
const MAX_LIST_ITEMS = 12;

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

function cleanString(value: unknown): string {
	return typeof value === "string" ? value.trim().slice(0, MAX_TEXT_LENGTH) : "";
}

function cleanList(value: unknown): string[] {
	if (!Array.isArray(value)) return [];
	return value.map(cleanString).filter(Boolean).slice(0, MAX_LIST_ITEMS);
}

function parseAnnotation(value: unknown): GifAnnotationDocument {
	const raw = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
	const decisionsRaw =
		raw.decisions && typeof raw.decisions === "object"
			? (raw.decisions as Record<string, unknown>)
			: {};
	const decisions = emptyDecisions();
	for (const field of GIF_ANNOTATION_FIELDS) {
		const decision = decisionsRaw[field];
		if (["pending", "accepted", "edited", "rejected"].includes(String(decision))) {
			decisions[field] = decision as SuggestionDecision;
		}
	}
	return {
		visibleText: cleanList(raw.visibleText),
		visualDescription: cleanString(raw.visualDescription),
		sequenceDescription: cleanString(raw.sequenceDescription),
		visualQueries: cleanList(raw.visualQueries),
		captionQueries: cleanList(raw.captionQueries),
		naturalQueries: cleanList(raw.naturalQueries),
		notes: cleanString(raw.notes),
		decisions,
	};
}

function parseStatus(value: unknown): GifAnnotationStatus {
	return value === "complete" || value === "skipped" ? value : "draft";
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

async function assertAdminAccess(): Promise<void> {
	const { allowed } = await checkAdminAccess();
	if (!allowed) throw new Error("Admin access required");
}

export const listGifAnnotations = createServerFn({ method: "GET" }).handler(async () => {
	await assertAdminAccess();
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
				annotation = parseAnnotation(JSON.parse(row.annotation_json));
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

interface SaveGifAnnotationInput {
	sha256: string;
	revision: number;
	status: GifAnnotationStatus;
	annotation: GifAnnotationDocument;
}

function validateSaveInput(input: SaveGifAnnotationInput): SaveGifAnnotationInput {
	if (!/^[a-f0-9]{64}$/.test(input.sha256)) throw new Error("Invalid GIF sha256");
	if (!Number.isInteger(input.revision) || input.revision < 0) throw new Error("Invalid revision");
	return {
		sha256: input.sha256,
		revision: input.revision,
		status: parseStatus(input.status),
		annotation: parseAnnotation(input.annotation),
	};
}

export const saveGifAnnotation = createServerFn({ method: "POST" })
	.inputValidator(validateSaveInput)
	.handler(async ({ data }) => {
		await assertAdminAccess();
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
