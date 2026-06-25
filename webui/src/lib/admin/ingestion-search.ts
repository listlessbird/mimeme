import { createParser, parseAsInteger, parseAsIsoDate, parseAsStringEnum } from "nuqs";

export const INGESTION_VIEWS = ["live", "completed", "failed", "all"] as const;
export type IngestionView = (typeof INGESTION_VIEWS)[number];

export const INGEST_STAGES = [
	"QUEUED",
	"DOWNLOADING",
	"PROCESSING",
	"ANNOTATING",
	"EMBEDDING",
	"COMPLETE",
	"DEDUPED",
] as const;
export type IngestStage = (typeof INGEST_STAGES)[number];

export const INGEST_TRIGGERS = ["scheduled", "manual"] as const;
export type IngestTrigger = (typeof INGEST_TRIGGERS)[number];

export const INGEST_OUTCOMES = ["ingested", "deduped"] as const;
export type IngestOutcome = (typeof INGEST_OUTCOMES)[number];

export const INGESTION_PAGE_SIZE = 50;
export const MAX_DATASET_LENGTH = 255;


export const parseAsDataset = createParser({
	parse: (value: string) => {
		const trimmed = value.trim();
		if (!trimmed || trimmed.length > MAX_DATASET_LENGTH) return null;
		return trimmed;
	},
	serialize: (value: string) => value,
});


export const parseAsPage = createParser({
	parse: (value: string) => {
		const n = Number.parseInt(value, 10);
		return Number.isFinite(n) && n >= 1 ? n : null;
	},
	serialize: (value: number) => String(value),
});


export const ingestionSearchParsers = {
	view: parseAsStringEnum<IngestionView>([...INGESTION_VIEWS]).withDefault("live"),
	stage: parseAsStringEnum<IngestStage>([...INGEST_STAGES]),
	trigger: parseAsStringEnum<IngestTrigger>([...INGEST_TRIGGERS]),
	source_id: parseAsInteger,
	dataset: parseAsDataset,
	outcome: parseAsStringEnum<IngestOutcome>([...INGEST_OUTCOMES]),
	from: parseAsIsoDate,
	to: parseAsIsoDate,
	page: parseAsPage.withDefault(1),
};

export function isLiveView(view: IngestionView): boolean {
	return view === "live";
}

export function pageToOffset(page: number, pageSize = INGESTION_PAGE_SIZE): number {
	return Math.max(0, (page - 1) * pageSize);
}
