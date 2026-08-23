import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardAction,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Field, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Kbd, KbdGroup } from "@/components/ui/kbd";
import { Progress } from "@/components/ui/progress";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { adminErrorMessage } from "@/lib/admin/api";
import {
	addSearchEvalCandidate,
	clearSearchEvalJudgment,
	saveSearchEvalJudgment,
	searchEvalJudgmentsQueryOptions,
	type SearchEvalJudgmentWorkspace,
} from "@/lib/admin/search-eval-api";
import { cn } from "@/lib/utils";
import { useHotkeys } from "@tanstack/react-hotkeys";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import {
	AlertTriangle,
	ArrowLeft,
	ArrowRight,
	ExternalLink,
	ImagePlus,
	Keyboard,
} from "lucide-react";
import {
	createContext,
	use,
	useCallback,
	useEffect,
	useMemo,
	useRef,
	useState,
	type ReactNode,
} from "react";

type Candidate = Omit<SearchEvalJudgmentWorkspace["candidates"][number], "grade"> & {
	grade: number | null;
};

interface JudgmentSessionValue {
	workspace: SearchEvalJudgmentWorkspace;
	candidates: Candidate[];
	activeCandidate: Candidate | null;
	activeIndex: number;
	isSaving: boolean;
	error: unknown;
	selectCandidate: (index: number) => void;
	moveCandidate: (direction: -1 | 1) => void;
	grade: (grade: number) => void;
	clearGrade: () => void;
}

const JudgmentSessionContext = createContext<JudgmentSessionValue | null>(null);

function useJudgmentSession(): JudgmentSessionValue {
	const session = use(JudgmentSessionContext);
	if (!session) throw new Error("Judgment controls must be inside JudgmentSession");
	return session;
}

/** Blind, keyboard-first relevance grading for one pooled query. */
export function SearchEvalJudgment({ queryId }: { queryId?: number }) {
	const judgment = useQuery(searchEvalJudgmentsQueryOptions(queryId));
	if (judgment.isPending) return <JudgmentSkeleton />;
	if (!judgment.data) {
		return (
			<Empty className="border">
				<EmptyHeader>
					<EmptyTitle>No candidates to judge</EmptyTitle>
					<EmptyDescription>
						Add a query and generate its candidates from the Queries page.
					</EmptyDescription>
				</EmptyHeader>
			</Empty>
		);
	}
	return (
		<JudgmentSession workspace={judgment.data}>
			<div className="flex flex-col gap-5">
				<JudgmentHeader />
				<JudgmentCanvas />
			</div>
		</JudgmentSession>
	);
}

function stableOrder(queryId: number, imageId: number): number {
	let value = (queryId * 1_000_003) ^ imageId;
	value = Math.imul(value ^ (value >>> 16), 0x45d9f3b);
	return value ^ (value >>> 16);
}

function JudgmentSession({
	workspace,
	children,
}: {
	workspace: SearchEvalJudgmentWorkspace;
	children: ReactNode;
}) {
	const queryClient = useQueryClient();
	const candidates = useMemo(
		() =>
			workspace.candidates
				.map((candidate) => ({
					candidate: { ...candidate, grade: candidate.grade ?? null },
					order: stableOrder(workspace.query.id, candidate.image_id),
				}))
				.sort((left, right) => left.order - right.order)
				.map(({ candidate }) => candidate),
		[workspace.candidates, workspace.query.id],
	);
	const firstUnjudged = Math.max(
		0,
		candidates.findIndex((candidate) => candidate.grade === null),
	);
	const [activeIndex, setActiveIndex] = useState(firstUnjudged);
	const save = useMutation({ mutationFn: saveSearchEvalJudgment });
	const clear = useMutation({ mutationFn: clearSearchEvalJudgment });

	useEffect(() => setActiveIndex(firstUnjudged), [firstUnjudged, workspace.query.id]);
	const activeCandidate = candidates[activeIndex] ?? null;
	const moveCandidate = useCallback(
		(direction: -1 | 1) => {
			setActiveIndex((current) => {
				if (!candidates.length) return 0;
				return (current + direction + candidates.length) % candidates.length;
			});
		},
		[candidates.length],
	);
	const grade = useCallback(
		(gradeValue: number) => {
			if (!activeCandidate || save.isPending || clear.isPending) return;
			void save
				.mutateAsync({
					data: {
						queryId: workspace.query.id,
						imageId: activeCandidate.image_id,
						grade: gradeValue,
						revision: activeCandidate.revision,
					},
				})
				.then((saved) => {
					queryClient.setQueriesData<SearchEvalJudgmentWorkspace>(
						{ queryKey: ["admin", "search-evals", "judgments"] },
						(current) =>
							current
								? {
										...current,
										candidates: current.candidates.map((candidate) =>
											candidate.image_id === saved.image_id
												? { ...candidate, grade: saved.grade, revision: saved.revision }
												: candidate,
										),
									}
								: current,
					);
					moveCandidate(1);
				})
				.catch(() => undefined);
		},
		[activeCandidate, clear.isPending, moveCandidate, queryClient, save, workspace.query.id],
	);
	const clearGrade = useCallback(() => {
		if (!activeCandidate || activeCandidate.grade === null || save.isPending || clear.isPending) {
			moveCandidate(1);
			return;
		}
		void clear
			.mutateAsync({
				data: {
					queryId: workspace.query.id,
					imageId: activeCandidate.image_id,
					revision: activeCandidate.revision,
				},
			})
			.then(() => {
				queryClient.setQueriesData<SearchEvalJudgmentWorkspace>(
					{ queryKey: ["admin", "search-evals", "judgments"] },
					(current) =>
						current
							? {
									...current,
									candidates: current.candidates.map((candidate) =>
										candidate.image_id === activeCandidate.image_id
											? { ...candidate, grade: null, revision: 0 }
											: candidate,
									),
								}
							: current,
				);
				moveCandidate(1);
			})
			.catch(() => undefined);
	}, [activeCandidate, clear, moveCandidate, queryClient, save.isPending, workspace.query.id]);

	useHotkeys(
		[
			{ hotkey: "0", callback: () => grade(0) },
			{ hotkey: "1", callback: () => grade(1) },
			{ hotkey: "2", callback: () => grade(2) },
			{ hotkey: "3", callback: () => grade(3) },
			{ hotkey: "ArrowLeft", callback: () => moveCandidate(-1) },
			{ hotkey: "ArrowRight", callback: () => moveCandidate(1) },
			{ hotkey: "P", callback: () => moveCandidate(-1) },
			{ hotkey: "N", callback: () => moveCandidate(1) },
			{ hotkey: "S", callback: clearGrade },
		],
		{
			enabled: Boolean(activeCandidate) && !save.isPending && !clear.isPending,
			ignoreInputs: true,
		},
	);

	const value = useMemo<JudgmentSessionValue>(
		() => ({
			workspace,
			candidates,
			activeCandidate,
			activeIndex,
			isSaving: save.isPending || clear.isPending,
			error: save.error ?? clear.error,
			selectCandidate: setActiveIndex,
			moveCandidate,
			grade,
			clearGrade,
		}),
		[
			workspace,
			candidates,
			activeCandidate,
			activeIndex,
			save.isPending,
			save.error,
			clear.isPending,
			clear.error,
			moveCandidate,
			grade,
			clearGrade,
		],
	);
	return <JudgmentSessionContext value={value}>{children}</JudgmentSessionContext>;
}

function JudgmentHeader() {
	const { workspace, candidates } = useJudgmentSession();
	const navigate = useNavigate();
	const judged = candidates.filter((candidate) => candidate.grade !== null).length;
	const progress = candidates.length ? (judged / candidates.length) * 100 : 0;
	return (
		<div className="flex flex-col gap-4 rounded-xl border bg-card p-4 shadow-sm md:p-5">
			<div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
				<div className="flex max-w-3xl flex-col gap-2">
					<div className="flex items-center gap-2">
						<Badge variant="outline">{workspace.query.intent}</Badge>
						<span className="text-xs text-muted-foreground">blind pool</span>
					</div>
					<p className="text-lg leading-snug font-semibold text-balance md:text-2xl">
						{workspace.query.text}
					</p>
				</div>
				<div className="flex gap-1">
					<Button
						variant="outline"
						size="icon-sm"
						aria-label="Previous query"
						disabled={!workspace.previous_query_id}
						onClick={() =>
							workspace.previous_query_id &&
							navigate({ to: "/admin/evals/judge", search: { query: workspace.previous_query_id } })
						}
					>
						<ArrowLeft />
					</Button>
					<Button
						variant="outline"
						size="icon-sm"
						aria-label="Next query"
						disabled={!workspace.next_query_id}
						onClick={() =>
							workspace.next_query_id &&
							navigate({ to: "/admin/evals/judge", search: { query: workspace.next_query_id } })
						}
					>
						<ArrowRight />
					</Button>
				</div>
			</div>
			<div className="flex items-center gap-3">
				<Progress
					value={progress}
					className="flex-1"
					aria-label={`${judged} of ${candidates.length} judged`}
				/>
				<span className="text-xs text-muted-foreground tabular-nums">
					{judged}/{candidates.length}
				</span>
			</div>
		</div>
	);
}

function JudgmentCanvas() {
	const { workspace, activeCandidate, activeIndex, candidates, grade, isSaving, error } =
		useJudgmentSession();
	if (!activeCandidate)
		return (
			<Empty className="border">
				<EmptyHeader>
					<EmptyTitle>This pool is empty</EmptyTitle>
					<EmptyDescription>
						Generate candidates for this query from the Queries page.
					</EmptyDescription>
				</EmptyHeader>
			</Empty>
		);
	return (
		<div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_22rem]">
			<div className="flex min-h-80 min-w-0 flex-col overflow-hidden rounded-xl border bg-muted/30 sm:min-h-[28rem] md:min-h-[38rem]">
				<div className="relative flex min-h-0 flex-1 items-center justify-center p-3">
					{activeCandidate.url ? (
						<img
							src={activeCandidate.url}
							alt={`Candidate meme ${activeCandidate.image_id}`}
							className="max-h-[72vh] w-auto max-w-full object-contain outline outline-1 -outline-offset-1 outline-black/10 dark:outline-white/10"
						/>
					) : (
						<span className="text-sm text-muted-foreground">Image unavailable</span>
					)}
					<Badge variant="secondary" className="absolute top-3 left-3">
						{activeIndex + 1} / {candidates.length}
					</Badge>
					<Button
						variant="secondary"
						size="icon-sm"
						nativeButton={false}
						className="absolute top-3 right-3"
						aria-label="Open image details"
						render={
							<Link
								to="/admin/images/$imageId"
								params={{ imageId: String(activeCandidate.image_id) }}
							/>
						}
					>
						<ExternalLink />
					</Button>
				</div>
				<CandidateFilmstrip />
			</div>
			<div className="flex flex-col gap-4">
				<Card className="@container">
					<CardHeader>
						<CardTitle>How well does it answer the query?</CardTitle>
						<CardDescription>Judge relevance, not image quality or personal taste.</CardDescription>
						<CardAction>
							<Keyboard />
						</CardAction>
					</CardHeader>
					<CardContent className="flex flex-col gap-5">
						<ToggleGroup
							variant="outline"
							value={activeCandidate.grade === null ? [] : [String(activeCandidate.grade)]}
							disabled={isSaving}
							onValueChange={([value]) => {
								const parsed = Number(value);
								if (Number.isInteger(parsed) && parsed >= 0 && parsed <= 3) grade(parsed);
							}}
							className="grid w-full grid-cols-1 gap-2 @md:grid-cols-2"
							spacing={2}
						>
							{GRADE_OPTIONS.map((option) => (
								<ToggleGroupItem
									key={option.grade}
									value={String(option.grade)}
									className="h-auto min-w-0 flex-col items-start gap-1 px-3 py-3 text-left"
								>
									<span className="flex w-full items-center justify-between">
										<span className="font-semibold">{option.label}</span>
										<Kbd>{option.grade}</Kbd>
									</span>
									<span className="w-full text-xs leading-4 font-normal text-pretty whitespace-normal text-muted-foreground">
										{option.description}
									</span>
								</ToggleGroupItem>
							))}
						</ToggleGroup>
						<div className="flex flex-wrap gap-x-4 gap-y-2 text-xs text-muted-foreground">
							<KbdGroup>
								<Kbd>N</Kbd>
								<span>next</span>
							</KbdGroup>
							<KbdGroup>
								<Kbd>P</Kbd>
								<span>previous</span>
							</KbdGroup>
							<KbdGroup>
								<Kbd>S</Kbd>
								<span>unjudge and skip</span>
							</KbdGroup>
						</div>
					</CardContent>
				</Card>
				<ManualCandidate queryId={workspace.query.id} />
				{error ? (
					<Alert variant="destructive">
						<AlertTriangle />
						<AlertTitle>Grade not saved</AlertTitle>
						<AlertDescription>
							{adminErrorMessage(error, "Try grading this meme again.")}
						</AlertDescription>
					</Alert>
				) : null}
			</div>
		</div>
	);
}

const GRADE_OPTIONS = [
	{ grade: 0, label: "Irrelevant", description: "Does not answer the need" },
	{ grade: 1, label: "Weak", description: "Related, but not useful" },
	{ grade: 2, label: "Relevant", description: "A useful result" },
	{ grade: 3, label: "Excellent", description: "Belongs near rank one" },
] as const;

function ManualCandidate({ queryId }: { queryId: number }) {
	const queryClient = useQueryClient();
	const [imageId, setImageId] = useState("");
	const add = useMutation({
		mutationFn: () => addSearchEvalCandidate({ data: { queryId, imageId: Number(imageId) } }),
		onSuccess: () => {
			setImageId("");
			return queryClient.invalidateQueries({ queryKey: ["admin", "search-evals", "judgments"] });
		},
	});
	return (
		<Card>
			<CardHeader>
				<CardTitle>Retrieval missed a good meme?</CardTitle>
				<CardDescription>Add it to this pool by image ID.</CardDescription>
			</CardHeader>
			<CardContent>
				<Field orientation="horizontal">
					<FieldLabel htmlFor="manual-image-id" className="sr-only">
						Image ID
					</FieldLabel>
					<Input
						id="manual-image-id"
						inputMode="numeric"
						value={imageId}
						onChange={(event) => setImageId(event.target.value)}
						placeholder="Image ID"
					/>
					<Button
						variant="outline"
						disabled={!Number.isInteger(Number(imageId)) || Number(imageId) <= 0 || add.isPending}
						onClick={() => add.mutate()}
					>
						<ImagePlus data-icon="inline-start" />
						Add
					</Button>
				</Field>
			</CardContent>
		</Card>
	);
}

function CandidateFilmstrip() {
	const { candidates, activeIndex, selectCandidate } = useJudgmentSession();
	const activeButtonRef = useRef<HTMLButtonElement>(null);
	useEffect(() => {
		activeButtonRef.current?.scrollIntoView({ block: "nearest", inline: "nearest" });
	}, [activeIndex]);
	return (
		<section className="min-w-0 border-t bg-card/80">
			<h2 className="sr-only">Candidate pool</h2>
			<ScrollArea className="w-full">
				<div className="flex w-max min-w-full gap-2 p-2 pb-3">
					{candidates.map((candidate, index) => (
						<Button
							key={candidate.image_id}
							ref={index === activeIndex ? activeButtonRef : undefined}
							variant="ghost"
							type="button"
							aria-label={`Select candidate ${index + 1}`}
							aria-current={index === activeIndex ? "true" : undefined}
							className={cn(
								"relative size-14 shrink-0 overflow-hidden rounded-none border bg-muted p-0 transition-[border-color,box-shadow,transform] outline-none focus-visible:ring-2 focus-visible:ring-ring active:scale-[0.96] sm:size-16",
								index === activeIndex && "border-foreground ring-1 ring-foreground",
							)}
							onClick={() => selectCandidate(index)}
						>
							{candidate.url ? (
								<img
									src={candidate.url}
									alt=""
									className="size-full object-cover outline outline-1 -outline-offset-1 outline-black/10 dark:outline-white/10"
									loading="lazy"
								/>
							) : null}
							{candidate.grade !== null ? (
								<Badge
									variant={candidate.grade >= 2 ? "default" : "secondary"}
									className="absolute right-1 bottom-1 min-w-5 px-1"
								>
									{candidate.grade}
								</Badge>
							) : null}
						</Button>
					))}
				</div>
				<ScrollBar orientation="horizontal" />
			</ScrollArea>
		</section>
	);
}

function JudgmentSkeleton() {
	return (
		<div className="flex flex-col gap-4">
			<Skeleton className="h-32" />
			<div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_22rem]">
				<Skeleton className="h-[38rem]" />
				<Skeleton className="h-80" />
			</div>
		</div>
	);
}
