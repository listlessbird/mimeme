import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardAction,
	CardContent,
	CardDescription,
	CardFooter,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import {
	Field,
	FieldContent,
	FieldDescription,
	FieldError,
	FieldGroup,
	FieldLabel,
} from "@/components/ui/field";
import {
	InputGroup,
	InputGroupAddon,
	InputGroupButton,
	InputGroupInput,
} from "@/components/ui/input-group";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import {
	emptyAnnotation,
	gifAnnotationSchema,
	gifAnnotationStatusSchema,
	gifAnnotationsQueryOptions,
	type GifAnnotationDocument,
	type GifAnnotationField,
	type GifAnnotationFormInput,
	type GifAnnotationItem,
	type GifAnnotationListResponse,
	type GifAnnotationStatus,
	saveGifAnnotation,
} from "@/lib/admin/gif-annotation";
import { buildEvalExport } from "@/lib/admin/gif-annotation-export";
import { zodResolver } from "@hookform/resolvers/zod";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
	AlertTriangle,
	ChevronLeft,
	ChevronRight,
	Download,
	ExternalLink,
	FileCheck2,
	Plus,
	RotateCcw,
	Save,
	SkipForward,
	Sparkles,
	Trash2,
} from "lucide-react";
import { parseAsString, useQueryState } from "nuqs";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Controller, useForm, useWatch, type Control } from "react-hook-form";
import { z } from "zod";

type SaveState = "idle" | "dirty" | "saving" | "saved" | "error";
type ListField = "visibleText" | "visualQueries" | "captionQueries" | "naturalQueries";
type TextField = "visualDescription" | "sequenceDescription";

const LIST_FIELDS: Array<{
	field: ListField;
	label: string;
	description: string;
	placeholder: string;
}> = [
	{
		field: "visibleText",
		label: "visible text",
		description: "Transcribe each distinct piece of on-screen text.",
		placeholder: "Add visible text",
	},
	{
		field: "visualQueries",
		label: "visual queries",
		description: "Queries someone could make from the visible content alone.",
		placeholder: "Add a visual query",
	},
	{
		field: "captionQueries",
		label: "caption queries",
		description: "Exact or near-exact searches for the overlaid text.",
		placeholder: "Add a caption query",
	},
	{
		field: "naturalQueries",
		label: "natural queries",
		description: "Phrases a person would genuinely type to find this GIF.",
		placeholder: "Add a natural query",
	},
];

const TEXT_FIELDS: Array<{
	field: TextField;
	label: string;
	description: string;
	placeholder: string;
}> = [
	{
		field: "visualDescription",
		label: "visual description",
		description: "Describe the people, objects, expressions, setting, and action.",
		placeholder: "What is visibly happening?",
	},
	{
		field: "sequenceDescription",
		label: "sequence",
		description: "Describe what changes from the start of the loop to the end.",
		placeholder: "How does the GIF progress?",
	},
];

const fallbackSchema = z.object({
	annotation: gifAnnotationSchema,
	status: gifAnnotationStatusSchema,
});

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

function withSuggestionDefaults(
	item: GifAnnotationItem,
	annotation: GifAnnotationDocument,
): GifAnnotationDocument {
	if (!item.suggestion) return annotation;
	const next = structuredClone(annotation);
	for (const field of [...LIST_FIELDS, ...TEXT_FIELDS].map((config) => config.field)) {
		const value = next[field];
		const empty = Array.isArray(value) ? value.length === 0 : value.length === 0;
		if (next.decisions[field] !== "pending" || !empty) continue;
		next[field] = suggestionValue(item, field) as never;
		next.decisions[field] = "accepted";
	}
	return next;
}

function fallbackKey(sha256: string): string {
	return `mimeme:gif-annotation:${sha256}`;
}

export function GifAnnotationWorkspace() {
	const queryClient = useQueryClient();
	const query = useQuery(gifAnnotationsQueryOptions());
	const data = query.data;
	const [activeSha, setActiveSha] = useQueryState("gif", parseAsString);
	const [status, setStatus] = useState<GifAnnotationStatus>("draft");
	const [revision, setRevision] = useState(0);
	const [saveState, setSaveState] = useState<SaveState>("idle");
	const [saveError, setSaveError] = useState<string | null>(null);
	const changeVersion = useRef(0);
	const saving = useRef<Promise<boolean> | null>(null);
	const loadedSha = useRef<string | null>(null);
	const form = useForm<GifAnnotationFormInput, unknown, GifAnnotationDocument>({
		resolver: zodResolver(gifAnnotationSchema),
		defaultValues: emptyAnnotation(),
		mode: "onBlur",
	});
	const watchedValues = useWatch({ control: form.control });

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

		const stored = window.localStorage.getItem(fallbackKey(item.sha256));
		let restored: z.infer<typeof fallbackSchema> | null = null;
		if (stored) {
			try {
				const fallback = fallbackSchema.safeParse(JSON.parse(stored));
				restored = fallback.success ? fallback.data : null;
			} catch {
				window.localStorage.removeItem(fallbackKey(item.sha256));
			}
		}
		form.reset(withSuggestionDefaults(item, restored?.annotation ?? item.annotation));
		setStatus(restored?.status ?? item.status);
		setRevision(item.revision);
		setSaveState(restored ? "dirty" : "idle");
		setSaveError(null);
		changeVersion.current += 1;
	}, [activeSha, form, item, setActiveSha]);

	useEffect(() => {
		if (!item || !form.formState.isDirty) return;
		const annotation = gifAnnotationSchema.safeParse(watchedValues);
		if (!annotation.success) return;
		window.localStorage.setItem(
			fallbackKey(item.sha256),
			JSON.stringify({ annotation: annotation.data, status }),
		);
	}, [form.formState.isDirty, item, status, watchedValues]);

	const markDirty = useCallback(() => {
		changeVersion.current += 1;
		setSaveState("dirty");
		setSaveError(null);
	}, []);

	const updateDecision = useCallback(
		(field: GifAnnotationField, value: string | string[]) => {
			if (!item) return;
			const suggested = suggestionValue(item, field);
			const empty = Array.isArray(value) ? value.every((entry) => !entry.trim()) : !value.trim();
			const matches =
				Array.isArray(value) && Array.isArray(suggested)
					? JSON.stringify(value) === JSON.stringify(suggested)
					: value === suggested;
			form.setValue(`decisions.${field}`, empty ? "rejected" : matches ? "accepted" : "edited", {
				shouldDirty: true,
			});
			markDirty();
		},
		[form, item, markDirty],
	);

	const persist = useCallback(
		async (
			annotation: GifAnnotationDocument,
			force = false,
			statusOverride?: GifAnnotationStatus,
		): Promise<boolean> => {
			if (!item || (!force && !["dirty", "error"].includes(saveState))) {
				return saveState !== "error";
			}
			if (saving.current) return saving.current;
			const savedVersion = changeVersion.current;
			const savedStatus = statusOverride ?? status;
			setSaveState("saving");
			const promise = saveGifAnnotation({
				data: { sha256: item.sha256, annotation, status: savedStatus, revision },
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
											annotation,
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
					form.reset(annotation);
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
		[form, item, queryClient, revision, saveState, status],
	);

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
		if (next) void setActiveSha(next.sha256);
	}, [activeIndex, data, setActiveSha]);

	const submitAnnotation = useCallback(
		async (annotation: GifAnnotationDocument, nextStatus: "complete" | "skipped") => {
			setStatus(nextStatus);
			changeVersion.current += 1;
			setSaveState("dirty");
			const saved = await persist(annotation, true, nextStatus);
			if (saved && nextStatus === "complete") navigateTo(activeIndex + 1);
		},
		[activeIndex, navigateTo, persist],
	);

	const retrySave = useCallback(() => {
		const annotation = gifAnnotationSchema.safeParse(form.getValues());
		if (annotation.success) void persist(annotation.data);
	}, [form, persist]);

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
	return (
		<div className="flex min-w-0 flex-col gap-4 pb-[env(safe-area-inset-bottom)] sm:gap-6">
			<header className="flex flex-col gap-3">
				<div className="flex items-start justify-between gap-3">
					<div className="flex min-w-0 flex-col gap-1">
						<h1 className="text-lg font-semibold">gif annotation</h1>
						<p className="text-sm text-muted-foreground">
							{data.completed} complete · {data.skipped} skipped · {data.total} total
						</p>
					</div>
					<Button variant="outline" size="icon-sm" onClick={() => downloadExport(data.items)}>
						<Download />
						<span className="sr-only">export annotations</span>
					</Button>
				</div>
				<Progress value={progress} aria-label={`${Math.round(progress)} percent complete`} />
			</header>

			<div className="grid min-w-0 gap-4 lg:grid-cols-[minmax(280px,0.8fr)_minmax(420px,1.2fr)] lg:gap-6">
				<Card className="gap-4 py-4 lg:sticky lg:top-4 lg:self-start">
					<CardHeader className="grid-cols-[1fr_auto] px-4 sm:px-6">
						<div className="flex min-w-0 flex-wrap items-center gap-2">
							<Badge variant="outline">
								{activeIndex + 1} / {data.total}
							</Badge>
							<Badge variant="secondary">{item.nFrames} frames</Badge>
							<Badge variant="secondary">{(item.durationMs / 1000).toFixed(1)}s</Badge>
						</div>
						<CardAction className="flex gap-1">
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
						</CardAction>
					</CardHeader>
					<CardContent className="px-0 sm:px-6">
						<div className="flex min-h-56 items-center justify-center overflow-hidden bg-black sm:rounded-md sm:border lg:min-h-96">
							<img
								key={item.sha256}
								src={item.gifUrl}
								alt={`GIF ${activeIndex + 1} for annotation`}
								className="max-h-[56svh] w-full object-contain lg:max-h-[68vh]"
							/>
						</div>
					</CardContent>
					<CardFooter className="flex-col items-stretch gap-2 px-4 sm:px-6">
						{/* oxlint-disable-next-line jsx-a11y/control-has-associated-label -- The rendered anchor below provides the control label. */}
						<Button
							variant="outline"
							size="sm"
							render={
								<a
									href={item.contactSheetUrl}
									target="_blank"
									rel="noreferrer"
									aria-label="Open sampled frames"
								>
									<ExternalLink data-icon="inline-start" />
									open sampled frames
								</a>
							}
							nativeButton={false}
						/>
						<p className="truncate font-mono text-[11px] text-muted-foreground" title={item.sha256}>
							{item.sha256}
						</p>
					</CardFooter>
				</Card>

				<form
					className="min-w-0"
					onSubmit={form.handleSubmit((annotation) => submitAnnotation(annotation, "complete"))}
				>
					<Card className="gap-0 py-0">
						<CardHeader className="border-b px-4 py-4 sm:px-6">
							<CardTitle>annotation</CardTitle>
							<CardDescription>
								Review the suggestion and describe how people would find this GIF.
							</CardDescription>
							<CardAction className="flex flex-col items-end gap-1">
								<Badge variant="secondary">{status}</Badge>
								<SaveIndicator state={saveState} error={saveError} onRetry={retrySave} />
							</CardAction>
						</CardHeader>

						<CardContent className="flex flex-col gap-6 px-4 py-5 sm:px-6">
							<SuggestionSummary item={item} />

							<FieldGroup>
								{LIST_FIELDS.map((config) => (
									<AnnotationListField
										key={config.field}
										control={form.control}
										config={config}
										decision={watchedValues.decisions?.[config.field] ?? "pending"}
										onChange={(value) => updateDecision(config.field, value)}
									/>
								))}

								{TEXT_FIELDS.map((config) => (
									<AnnotationTextField
										key={config.field}
										control={form.control}
										config={config}
										decision={watchedValues.decisions?.[config.field] ?? "pending"}
										onChange={(value) => updateDecision(config.field, value)}
									/>
								))}

								<Controller
									control={form.control}
									name="notes"
									render={({ field, fieldState }) => (
										<Field data-invalid={fieldState.invalid}>
											<FieldLabel htmlFor="annotation-notes">notes</FieldLabel>
											<FieldDescription>
												Optional context for reviewers or future cleanup.
											</FieldDescription>
											<Textarea
												{...field}
												id="annotation-notes"
												rows={3}
												maxLength={2_000}
												aria-invalid={fieldState.invalid}
												onChange={(event) => {
													field.onChange(event);
													markDirty();
												}}
											/>
											<FieldError errors={[fieldState.error]} />
										</Field>
									)}
								/>
							</FieldGroup>
						</CardContent>

						<CardFooter className="sticky bottom-0 grid grid-cols-2 gap-2 border-t bg-card/95 px-4 py-4 backdrop-blur sm:flex sm:justify-end sm:px-6">
							<Button
								className="w-full sm:w-auto"
								type="button"
								variant="outline"
								onClick={() =>
									void form.handleSubmit((annotation) => submitAnnotation(annotation, "skipped"))()
								}
								disabled={saveState === "saving"}
							>
								<SkipForward data-icon="inline-start" />
								skip
							</Button>
							<Button className="w-full sm:w-auto" type="submit" disabled={saveState === "saving"}>
								{saveState === "saving" ? (
									<Spinner data-icon="inline-start" />
								) : (
									<Save data-icon="inline-start" />
								)}
								{saveState === "saving" ? "saving" : "submit"}
							</Button>
						</CardFooter>
					</Card>
				</form>
			</div>

			<Button className="w-full sm:ml-auto sm:w-auto" variant="outline" onClick={setNextPending}>
				<SkipForward data-icon="inline-start" />
				next pending
			</Button>
		</div>
	);
}

function AnnotationListField({
	control,
	config,
	decision,
	onChange,
}: {
	control: Control<GifAnnotationFormInput, unknown, GifAnnotationDocument>;
	config: (typeof LIST_FIELDS)[number];
	decision: GifAnnotationDocument["decisions"][GifAnnotationField];
	onChange: (value: string[]) => void;
}) {
	return (
		<Controller
			control={control}
			name={config.field}
			render={({ field, fieldState }) => {
				const values = field.value;
				return (
					<Field data-invalid={fieldState.invalid}>
						<div className="flex items-start justify-between gap-3">
							<FieldContent>
								<FieldLabel>{config.label}</FieldLabel>
								<FieldDescription>{config.description}</FieldDescription>
							</FieldContent>
							<Badge variant="outline">{decision}</Badge>
						</div>
						<div className="flex flex-col gap-2">
							{values.map((value, index) => (
								<InputGroup key={`${config.field}-${index}`}>
									<InputGroupInput
										value={value}
										maxLength={2_000}
										aria-label={`${config.label} ${index + 1}`}
										aria-invalid={fieldState.invalid}
										placeholder={config.placeholder}
										onChange={(event) => {
											const next = values.with(index, event.target.value);
											field.onChange(next);
											onChange(next);
										}}
									/>
									<InputGroupAddon align="inline-end">
										<InputGroupButton
											size="icon-xs"
											onClick={() => {
												const next = values.filter((_, itemIndex) => itemIndex !== index);
												field.onChange(next);
												onChange(next);
											}}
											aria-label={`Remove ${config.label} ${index + 1}`}
										>
											<Trash2 />
										</InputGroupButton>
									</InputGroupAddon>
								</InputGroup>
							))}
							<Button
								type="button"
								variant="outline"
								size="sm"
								disabled={values.length >= 12}
								onClick={() => {
									const next = [...values, ""];
									field.onChange(next);
									onChange(next);
								}}
							>
								<Plus data-icon="inline-start" />
								add {config.field === "visibleText" ? "text" : "query"}
							</Button>
						</div>
						<FieldError errors={[fieldState.error]} />
					</Field>
				);
			}}
		/>
	);
}

function AnnotationTextField({
	control,
	config,
	decision,
	onChange,
}: {
	control: Control<GifAnnotationFormInput, unknown, GifAnnotationDocument>;
	config: (typeof TEXT_FIELDS)[number];
	decision: GifAnnotationDocument["decisions"][GifAnnotationField];
	onChange: (value: string) => void;
}) {
	return (
		<Controller
			control={control}
			name={config.field}
			render={({ field, fieldState }) => (
				<Field data-invalid={fieldState.invalid}>
					<div className="flex items-start justify-between gap-3">
						<FieldContent>
							<FieldLabel htmlFor={`annotation-${config.field}`}>{config.label}</FieldLabel>
							<FieldDescription>{config.description}</FieldDescription>
						</FieldContent>
						<Badge variant="outline">{decision}</Badge>
					</div>
					<Textarea
						{...field}
						id={`annotation-${config.field}`}
						rows={4}
						maxLength={2_000}
						placeholder={config.placeholder}
						aria-invalid={fieldState.invalid}
						onChange={(event) => {
							field.onChange(event);
							onChange(event.target.value);
						}}
					/>
					<FieldError errors={[fieldState.error]} />
				</Field>
			)}
		/>
	);
}

function SuggestionSummary({ item }: { item: GifAnnotationItem }) {
	if (!item.suggestion) {
		return (
			<Alert>
				<AlertTriangle />
				<AlertTitle>suggestion pending</AlertTitle>
				<AlertDescription>
					Human annotation can continue while this item is generated.
				</AlertDescription>
			</Alert>
		);
	}
	return (
		<div className="flex flex-col gap-3">
			<div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
				<Sparkles />
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
			<Button variant="destructive" size="xs" onClick={onRetry} title={error ?? undefined}>
				<RotateCcw data-icon="inline-start" />
				retry
			</Button>
		);
	}
	return (
		<div className="flex items-center gap-1 text-xs text-muted-foreground" aria-live="polite">
			{state === "saving" ? <Spinner /> : state === "saved" ? <FileCheck2 /> : <Save />}
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
			<div className="grid gap-6 lg:grid-cols-2">
				<Skeleton className="min-h-96 w-full" />
				<Skeleton className="min-h-[720px] w-full" />
			</div>
		</div>
	);
}
