import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
	Field,
	FieldContent,
	FieldDescription,
	FieldGroup,
	FieldLabel,
} from "@/components/ui/field";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import {
	emptyAnnotation,
	gifAnnotationsQueryOptions,
	type GifAnnotationDocument,
	type GifAnnotationField,
	type GifAnnotationItem,
	type GifAnnotationListResponse,
	type GifAnnotationStatus,
	saveGifAnnotation,
} from "@/lib/admin/gif-annotation";
import { buildEvalExport } from "@/lib/admin/gif-annotation-export";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
	AlertTriangle,
	ChevronLeft,
	ChevronRight,
	Download,
	ExternalLink,
	FileCheck2,
	RotateCcw,
	Save,
	SkipForward,
	Sparkles,
} from "lucide-react";
import { parseAsString, useQueryState } from "nuqs";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type SaveState = "idle" | "dirty" | "saving" | "saved" | "error";

const FIELD_CONFIG: Array<{
	field: GifAnnotationField;
	label: string;
	description: string;
	kind: "text" | "list";
}> = [
	{
		field: "visibleText",
		label: "visible text",
		description: "one transcription per line",
		kind: "list",
	},
	{
		field: "visualDescription",
		label: "visual description",
		description: "people, objects, expressions, and action",
		kind: "text",
	},
	{
		field: "sequenceDescription",
		label: "sequence",
		description: "what changes through the loop",
		kind: "text",
	},
	{
		field: "visualQueries",
		label: "visual queries",
		description: "queries based only on visible content",
		kind: "list",
	},
	{
		field: "captionQueries",
		label: "caption queries",
		description: "exact or near-exact overlaid text",
		kind: "list",
	},
	{
		field: "naturalQueries",
		label: "natural queries",
		description: "phrases you would genuinely search",
		kind: "list",
	},
];

function suggestionValue(item: GifAnnotationItem, field: GifAnnotationField): string | string[] {
	const suggestion = item.suggestion;
	if (!suggestion) return field.endsWith("Queries") || field === "visibleText" ? [] : "";
	switch (field) {
		case "visibleText":
			return suggestion.visible_text;
		case "visualDescription":
			return suggestion.visual_description;
		case "sequenceDescription":
			return suggestion.sequence_description;
		case "visualQueries":
			return suggestion.suggested_visual_queries;
		case "captionQueries":
			return suggestion.suggested_caption_queries;
		case "naturalQueries":
			return suggestion.suggested_natural_queries;
	}
}

function textValue(value: string | string[]): string {
	return Array.isArray(value) ? value.join("\n") : value;
}

function fieldTextValues(annotation: GifAnnotationDocument): Record<GifAnnotationField, string> {
	return Object.fromEntries(
		FIELD_CONFIG.map(({ field }) => [field, textValue(annotation[field])]),
	) as Record<GifAnnotationField, string>;
}

function parseLines(value: string): string[] {
	return value
		.split("\n")
		.map((line) => line.trim())
		.filter(Boolean);
}

function parseQueries(value: string): string[] {
	return value
		.split(/[\n,]/)
		.map((query) => query.trim())
		.filter(Boolean);
}

function isEmptyValue(value: string | string[]): boolean {
	return Array.isArray(value) ? value.length === 0 : value.trim().length === 0;
}

function valuesMatch(left: string | string[], right: string | string[]): boolean {
	return textValue(left).trim() === textValue(right).trim();
}

function withSuggestionDefaults(
	item: GifAnnotationItem,
	annotation: GifAnnotationDocument,
): GifAnnotationDocument {
	if (!item.suggestion) return annotation;
	const next = structuredClone(annotation);
	for (const { field } of FIELD_CONFIG) {
		if (next.decisions[field] !== "pending" || !isEmptyValue(next[field])) continue;
		next[field] = suggestionValue(item, field) as never;
		next.decisions[field] = "accepted";
	}
	return next;
}

function fallbackKey(sha256: string): string {
	return `mimeme:gif-annotation:${sha256}`;
}

function loadFallback(item: GifAnnotationItem) {
	if (typeof window === "undefined") return null;
	const stored = window.localStorage.getItem(fallbackKey(item.sha256));
	if (!stored) return null;
	try {
		return JSON.parse(stored) as {
			annotation: GifAnnotationDocument;
			status: GifAnnotationStatus;
		};
	} catch {
		return null;
	}
}

export function GifAnnotationWorkspace() {
	const queryClient = useQueryClient();
	const query = useQuery(gifAnnotationsQueryOptions());
	const data = query.data;
	const [activeSha, setActiveSha] = useQueryState("gif", parseAsString);
	const [draft, setDraft] = useState<GifAnnotationDocument>(emptyAnnotation);
	const [fieldText, setFieldText] = useState<Record<GifAnnotationField, string>>(() =>
		fieldTextValues(emptyAnnotation()),
	);
	const [status, setStatus] = useState<GifAnnotationStatus>("draft");
	const [revision, setRevision] = useState(0);
	const [saveState, setSaveState] = useState<SaveState>("idle");
	const [saveError, setSaveError] = useState<string | null>(null);
	const changeVersion = useRef(0);
	const saving = useRef<Promise<boolean> | null>(null);
	const loadedSha = useRef<string | null>(null);

	const activeIndex = useMemo(() => {
		if (!data?.items.length) return -1;
		const requested = activeSha ? data.items.findIndex((item) => item.sha256 === activeSha) : -1;
		if (requested >= 0) return requested;
		const firstPending = data.items.findIndex((item) => item.status === "draft");
		return firstPending >= 0 ? firstPending : 0;
	}, [activeSha, data]);
	const item = activeIndex >= 0 ? data?.items[activeIndex] : undefined;

	useEffect(() => {
		if (!item || loadedSha.current === item.sha256) return;
		loadedSha.current = item.sha256;
		if (activeSha !== item.sha256) void setActiveSha(item.sha256, { history: "replace" });
		const fallback = loadFallback(item);
		const nextDraft = withSuggestionDefaults(item, fallback?.annotation ?? item.annotation);
		setDraft(nextDraft);
		setFieldText(fieldTextValues(nextDraft));
		setStatus(fallback?.status ?? item.status);
		setRevision(item.revision);
		setSaveState(fallback ? "dirty" : "idle");
		setSaveError(null);
		changeVersion.current += 1;
	}, [activeSha, item, setActiveSha]);

	const updateDraft = useCallback(
		(next: GifAnnotationDocument | ((current: GifAnnotationDocument) => GifAnnotationDocument)) => {
			setDraft((current) => (typeof next === "function" ? next(current) : next));
			changeVersion.current += 1;
			setSaveState("dirty");
			setSaveError(null);
		},
		[],
	);

	const persist = useCallback(
		async (force = false, statusOverride?: GifAnnotationStatus): Promise<boolean> => {
			if (!item || (!force && !["dirty", "error"].includes(saveState))) {
				return saveState !== "error";
			}
			if (saving.current) return saving.current;
			const savedVersion = changeVersion.current;
			const savedStatus = statusOverride ?? status;
			setSaveState("saving");
			const promise = saveGifAnnotation({
				data: { sha256: item.sha256, annotation: draft, status: savedStatus, revision },
			})
				.then((result) => {
					setRevision(result.revision);
					queryClient.setQueryData<GifAnnotationListResponse>(
						gifAnnotationsQueryOptions().queryKey,
						(current) => {
							if (!current) return current;
							const items = current.items.map((candidate) =>
								candidate.sha256 === item.sha256
									? {
											...candidate,
											annotation: draft,
											status: savedStatus,
											revision: result.revision,
											updatedAt: result.updatedAt,
										}
									: candidate,
							);
							return {
								...current,
								items,
								completed: items.filter((candidate) => candidate.status === "complete").length,
								skipped: items.filter((candidate) => candidate.status === "skipped").length,
							};
						},
					);
					window.localStorage.removeItem(fallbackKey(item.sha256));
					setSaveState(changeVersion.current === savedVersion ? "saved" : "dirty");
					return true;
				})
				.catch((cause: unknown) => {
					setSaveState("error");
					setSaveError(cause instanceof Error ? cause.message : "annotation save failed");
					return false;
				})
				.finally(() => {
					saving.current = null;
				});
			saving.current = promise;
			return promise;
		},
		[draft, item, queryClient, revision, saveState, status],
	);

	useEffect(() => {
		if (!item || saveState !== "dirty") return;
		window.localStorage.setItem(
			fallbackKey(item.sha256),
			JSON.stringify({ annotation: draft, status }),
		);
	}, [draft, item, saveState, status]);

	const navigateTo = useCallback(
		(nextIndex: number) => {
			if (!data?.items[nextIndex]) return;
			void setActiveSha(data.items[nextIndex].sha256);
		},
		[data, setActiveSha],
	);

	const setNextPending = useCallback(() => {
		if (!data || activeIndex < 0) return;
		const after = data.items
			.slice(activeIndex + 1)
			.find((candidate) => candidate.status === "draft");
		const before = data.items
			.slice(0, activeIndex)
			.find((candidate) => candidate.status === "draft");
		const next = after ?? before;
		if (!next) return;
		void setActiveSha(next.sha256);
	}, [activeIndex, data, setActiveSha]);

	const exportAnnotations = useCallback(() => {
		const current = queryClient.getQueryData<GifAnnotationListResponse>(
			gifAnnotationsQueryOptions().queryKey,
		);
		if (current) downloadExport(current.items);
	}, [queryClient]);

	const submitAnnotation = useCallback(
		async (nextStatus: "complete" | "skipped") => {
			setStatus(nextStatus);
			changeVersion.current += 1;
			setSaveState("dirty");
			const saved = await persist(true, nextStatus);
			if (saved && nextStatus === "complete") navigateTo(activeIndex + 1);
		},
		[activeIndex, navigateTo, persist],
	);

	if (query.isPending) return <WorkspaceSkeleton />;
	if (query.error || !data) {
		return (
			<Alert variant="destructive">
				<AlertTriangle />
				<AlertTitle>gif annotations unavailable</AlertTitle>
				<AlertDescription>
					{query.error instanceof Error ? query.error.message : "dataset failed to load"}
				</AlertDescription>
			</Alert>
		);
	}
	if (!item) return null;

	const progress = data.total ? (data.completed / data.total) * 100 : 0;
	const setField = (field: GifAnnotationField, value: string) => {
		setFieldText((current) => ({ ...current, [field]: value }));
		const kind = FIELD_CONFIG.find((config) => config.field === field)?.kind;
		const parsed =
			kind === "list" ? (field === "visibleText" ? parseLines(value) : parseQueries(value)) : value;
		const suggested = suggestionValue(item, field);
		updateDraft((current) => ({
			...current,
			[field]: parsed,
			decisions: {
				...current.decisions,
				[field]: isEmptyValue(parsed)
					? "rejected"
					: valuesMatch(parsed, suggested)
						? "accepted"
						: "edited",
			},
		}));
	};
	return (
		<div className="flex min-w-0 flex-col gap-5 pb-[env(safe-area-inset-bottom)]">
			<header className="flex flex-col gap-3 border-b pb-4">
				<div className="flex flex-wrap items-start justify-between gap-4">
					<div className="flex flex-col gap-1">
						<h1 className="text-lg font-semibold">gif annotation</h1>
						<div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
							<span>{data.completed} complete</span>
							<span aria-hidden="true">/</span>
							<span>{data.skipped} skipped</span>
							<span aria-hidden="true">/</span>
							<span>{data.total} total</span>
						</div>
					</div>
					<div className="grid w-full grid-cols-2 gap-2 sm:flex sm:w-auto sm:flex-wrap">
						<Button
							className="w-full sm:w-auto"
							variant="outline"
							size="sm"
							onClick={exportAnnotations}
						>
							<Download data-icon="inline-start" />
							export
						</Button>
						<Button className="w-full sm:w-auto" size="sm" onClick={setNextPending}>
							<SkipForward data-icon="inline-start" />
							next pending
						</Button>
					</div>
				</div>
				<Progress value={progress} aria-label={`${Math.round(progress)} percent complete`} />
			</header>

			<div className="grid min-w-0 gap-6 xl:grid-cols-[minmax(320px,0.9fr)_minmax(460px,1.1fr)]">
				<section className="flex min-w-0 flex-col gap-4 xl:sticky xl:top-4 xl:self-start">
					<div className="flex items-center justify-between gap-3">
						<div className="flex flex-wrap items-center gap-2">
							<Badge variant="outline">
								{activeIndex + 1} / {data.total}
							</Badge>
							<Badge variant="secondary">{item.nFrames} frames</Badge>
							<Badge variant="secondary">{(item.durationMs / 1000).toFixed(1)}s</Badge>
						</div>
						<div className="flex items-center gap-1">
							<Button
								variant="outline"
								size="icon-sm"
								disabled={activeIndex <= 0}
								onClick={() => navigateTo(activeIndex - 1)}
								aria-label="Previous GIF"
							>
								<ChevronLeft />
							</Button>
							<Button
								variant="outline"
								size="icon-sm"
								disabled={activeIndex >= data.total - 1}
								onClick={() => navigateTo(activeIndex + 1)}
								aria-label="Next GIF"
							>
								<ChevronRight />
							</Button>
						</div>
					</div>
					<div className="flex min-h-[240px] items-center justify-center overflow-hidden rounded-md border bg-black sm:min-h-[460px]">
						<img
							key={item.sha256}
							src={item.gifUrl}
							alt={`GIF ${activeIndex + 1} for annotation`}
							className="max-h-[68vh] w-full object-contain"
						/>
					</div>
					<a
						href={item.contactSheetUrl}
						target="_blank"
						rel="noreferrer"
						className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
					>
						<ExternalLink className="size-3.5" />
						open sampled frames
					</a>
					<div className="font-mono text-[11px] break-all text-muted-foreground">{item.sha256}</div>
				</section>

				<section className="-mx-4 min-w-0 border-y bg-card p-4 sm:mx-0 sm:rounded-md sm:border sm:p-5">
					<div className="flex flex-wrap items-center justify-end gap-3 border-b pb-4">
						<Badge variant="secondary">{status}</Badge>
						<SaveIndicator state={saveState} error={saveError} onRetry={() => void persist()} />
					</div>

					{item.suggestion ? (
						<div className="mt-4 flex flex-col gap-2 text-xs text-muted-foreground">
							<div className="flex flex-wrap items-center gap-2">
								<Sparkles className="size-3.5" />
								<span>{item.suggestionModel}</span>
								{item.suggestion.supporting_frame_numbers.map((frame) => (
									<Badge key={frame} variant="outline">
										frame {frame}
									</Badge>
								))}
							</div>
							{item.suggestion.uncertainty ? (
								<Alert>
									<AlertTriangle />
									<AlertTitle>model uncertainty</AlertTitle>
									<AlertDescription>{item.suggestion.uncertainty}</AlertDescription>
								</Alert>
							) : null}
						</div>
					) : (
						<Alert className="mt-4">
							<AlertTriangle />
							<AlertTitle>suggestion pending</AlertTitle>
							<AlertDescription>
								human annotation can continue while this item is generated.
							</AlertDescription>
						</Alert>
					)}

					<FieldGroup className="mt-6">
						{FIELD_CONFIG.map((config) => (
							<SuggestionField
								key={config.field}
								config={config}
								value={fieldText[config.field]}
								decision={draft.decisions[config.field]}
								onChange={(value) => setField(config.field, value)}
							/>
						))}
						<Field>
							<FieldLabel htmlFor="annotation-notes">notes</FieldLabel>
							<FieldContent>
								<Textarea
									id="annotation-notes"
									value={draft.notes}
									onChange={(event) =>
										updateDraft((current) => ({ ...current, notes: event.target.value }))
									}
									rows={3}
								/>
							</FieldContent>
						</Field>
					</FieldGroup>
					<div className="mt-6 grid grid-cols-2 gap-2 border-t pt-4 sm:flex sm:justify-end">
						<Button
							className="w-full sm:w-auto"
							type="button"
							variant="outline"
							onClick={() => void submitAnnotation("skipped")}
							disabled={saveState === "saving"}
						>
							<SkipForward data-icon="inline-start" />
							skip GIF
						</Button>
						<Button
							className="w-full sm:w-auto"
							type="button"
							onClick={() => void submitAnnotation("complete")}
							disabled={saveState === "saving"}
						>
							<Save data-icon="inline-start" />
							submit annotation
						</Button>
					</div>
				</section>
			</div>
		</div>
	);
}

function SuggestionField({
	config,
	value,
	decision,
	onChange,
}: {
	config: (typeof FIELD_CONFIG)[number];
	value: string;
	decision: GifAnnotationDocument["decisions"][GifAnnotationField];
	onChange: (value: string) => void;
}) {
	return (
		<Field>
			<div className="flex flex-wrap items-start justify-between gap-3">
				<div className="flex flex-col gap-1">
					<FieldLabel htmlFor={`annotation-${config.field}`}>{config.label}</FieldLabel>
					<FieldDescription>{config.description}</FieldDescription>
				</div>
				<Badge variant="outline">{decision}</Badge>
			</div>
			<FieldContent>
				<Textarea
					id={`annotation-${config.field}`}
					value={value}
					onChange={(event) => onChange(event.target.value)}
					rows={config.kind === "list" ? 3 : 4}
					placeholder={
						config.kind === "list"
							? config.field === "visibleText"
								? "One transcription per line"
								: "One query per line, or separate queries with commas"
							: undefined
					}
				/>
			</FieldContent>
		</Field>
	);
}

function SaveIndicator({
	state,
	error,
	onRetry,
}: {
	state: SaveState;
	error: string | null;
	onRetry: () => void;
}) {
	if (state === "error") {
		return (
			<Button variant="destructive" size="sm" onClick={onRetry} title={error ?? undefined}>
				<RotateCcw data-icon="inline-start" />
				retry save
			</Button>
		);
	}
	return (
		<div className="flex items-center gap-2 text-xs text-muted-foreground" aria-live="polite">
			{state === "saved" ? <FileCheck2 className="size-3.5" /> : <Save className="size-3.5" />}
			<span>{state === "idle" ? "unchanged" : state}</span>
		</div>
	);
}

function downloadExport(items: GifAnnotationItem[]) {
	const contents = JSON.stringify(buildEvalExport(items), null, 2) + "\n";
	const url = URL.createObjectURL(new Blob([contents], { type: "application/json" }));
	const link = document.createElement("a");
	link.href = url;
	link.download = "gif-eval-set.json";
	link.click();
	URL.revokeObjectURL(url);
}

function WorkspaceSkeleton() {
	return (
		<div className="flex flex-col gap-5">
			<Skeleton className="h-16 w-full" />
			<div className="grid gap-6 xl:grid-cols-2">
				<Skeleton className="min-h-[520px] w-full" />
				<Skeleton className="min-h-[720px] w-full" />
			</div>
		</div>
	);
}
