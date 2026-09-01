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
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
	DialogTrigger,
} from "@/components/ui/dialog";
import {
	Empty,
	EmptyContent,
	EmptyDescription,
	EmptyHeader,
	EmptyMedia,
	EmptyTitle,
} from "@/components/ui/empty";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import {
	Select,
	SelectContent,
	SelectGroup,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Spinner } from "@/components/ui/spinner";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { adminErrorMessage } from "@/lib/admin/api";
import {
	createSearchEvalQuery,
	createSearchEvalExperiment,
	disableSearchEvalQuery,
	poolSearchEvalQuery,
	SEARCH_EVAL_INTENTS,
	searchEvalOverviewQueryOptions,
	setSearchEvalBaseline,
	type SearchEvalIntent,
	type SearchEvalOverview as OverviewData,
	type SearchEvalQuery,
	type SearchEvalQuerySource,
	type SearchEvalRun,
	type SearchEvalRecipe,
	type SearchEvalRecipeId,
} from "@/lib/admin/search-eval-api";
import { cn } from "@/lib/utils";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import {
	AlertTriangle,
	ArrowRight,
	Check,
	CirclePlus,
	Database,
	Gauge,
	Images,
	Play,
	RefreshCw,
	Scale,
	Trash2,
} from "lucide-react";
import { useState, type ReactNode } from "react";

const OVERVIEW_QUERY_KEY = ["admin", "search-evals"] as const;
const RUN_DATE_FORMATTER = new Intl.DateTimeFormat(undefined, {
	dateStyle: "medium",
	timeStyle: "short",
});

/** Dashboard for the persistent Core Search benchmark. */
export function SearchEvalOverview() {
	const overview = useQuery(searchEvalOverviewQueryOptions());
	if (overview.isPending) return <OverviewSkeleton />;
	if (!overview.data) return null;
	return <OverviewDashboard data={overview.data} />;
}

/** Query management with candidate discovery kept alongside each query. */
export function SearchEvalQueries() {
	const overview = useQuery(searchEvalOverviewQueryOptions());
	if (overview.isPending) return <ListSkeleton />;
	if (!overview.data) return null;
	return <QueriesWorkspace data={overview.data} />;
}

/** Evaluation history and controls for running the current search system. */
export function SearchEvalRuns() {
	const overview = useQuery(searchEvalOverviewQueryOptions());
	if (overview.isPending) return <ListSkeleton />;
	if (!overview.data) return null;
	return <RunsWorkspace data={overview.data} />;
}

function OverviewDashboard({ data }: { data: OverviewData }) {
	const activeQueries = data.queries.filter((query) => query.status === "active");
	const baseline = data.recent_runs.find((run) => run.id === data.baseline_run_id);
	const latestCandidate = data.recent_runs.find(
		(run) => run.status === "complete" && run.id !== data.baseline_run_id,
	);
	const needsJudgments = data.recent_runs.find((run) => run.status === "needs_judgments");
	const coverage = data.candidate_count
		? Math.round((data.judgment_count / data.candidate_count) * 100)
		: 0;
	const benchmarkReady =
		activeQueries.length > 0 &&
		activeQueries.every((query) => query.candidate_count > 0 && query.relevant_count > 0) &&
		data.unjudged_count === 0;

	return (
		<div className="flex flex-col gap-6">
			<div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
				<SummaryCard icon={Database} label="Active queries" value={data.active_query_count} />
				<SummaryCard icon={Images} label="Candidates" value={data.candidate_count} />
				<SummaryCard icon={Check} label="Judgment coverage" value={`${coverage}%`} />
				<SummaryCard
					icon={Gauge}
					label="Active baseline"
					value={baseline?.metrics ? baseline.metrics.ndcg_at_10.toFixed(3) : "Not set"}
					detail={baseline?.metrics ? "nDCG@10" : undefined}
				/>
			</div>

			{needsJudgments ? (
				<Alert>
					<AlertTriangle />
					<AlertTitle>{needsJudgments.missing_judgments} new results need judgments</AlertTitle>
					<AlertDescription>
						<p>Grade the new results before this evaluation can be scored.</p>
						<Button size="sm" nativeButton={false} render={<Link to="/admin/evals/judge" />}>
							Review judgments <ArrowRight data-icon="inline-end" />
						</Button>
					</AlertDescription>
				</Alert>
			) : null}

			{benchmarkReady ? null : <BenchmarkSetup data={data} />}

			<div className="grid gap-4 lg:grid-cols-2">
				<RunSummary
					title="Baseline"
					description="The reference run for Core Search."
					run={baseline}
					empty="Set a complete run as the baseline."
					footer={<ViewRunsButton />}
				/>
				<RunSummary
					title="Latest candidate"
					description="The latest complete run that is not the baseline."
					run={latestCandidate}
					empty="Create another evaluation to compare it with the baseline."
					footer={<ViewRunsButton />}
				/>
			</div>
		</div>
	);
}

function SummaryCard({
	icon: Icon,
	label,
	value,
	detail,
}: {
	icon: typeof Database;
	label: string;
	value: string | number;
	detail?: string;
}) {
	return (
		<Card className="gap-4 py-5">
			<CardHeader className="px-5">
				<CardDescription>{label}</CardDescription>
				<CardAction>
					<Icon className="size-4 text-muted-foreground" />
				</CardAction>
			</CardHeader>
			<CardContent className="flex items-baseline gap-2 px-5">
				<span className="text-2xl font-semibold tracking-tight tabular-nums">{value}</span>
				{detail ? <span className="text-xs text-muted-foreground">{detail}</span> : null}
			</CardContent>
		</Card>
	);
}

function BenchmarkSetup({ data }: { data: OverviewData }) {
	const activeQueries = data.queries.filter((query) => query.status === "active");
	const hasQueries = activeQueries.length > 0;
	const poolsReady = hasQueries && activeQueries.every((query) => query.candidate_count > 0);
	const judgmentsReady =
		poolsReady &&
		data.unjudged_count === 0 &&
		activeQueries.every((query) => query.relevant_count > 0);
	const nextAction = !hasQueries
		? { label: "Add queries", to: "/admin/evals/queries" as const }
		: !poolsReady
			? { label: "Generate candidates", to: "/admin/evals/queries" as const }
			: { label: "Continue judging", to: "/admin/evals/judge" as const };

	return (
		<Card>
			<CardHeader>
				<CardTitle>Finish setting up the benchmark</CardTitle>
				<CardDescription>
					Runs become meaningful once every active query has candidates and at least one relevant
					result.
				</CardDescription>
			</CardHeader>
			<CardContent className="grid gap-3 sm:grid-cols-3">
				<SetupItem complete={hasQueries} label={`${activeQueries.length} queries added`} />
				<SetupItem complete={poolsReady} label="Candidate pools generated" />
				<SetupItem
					complete={judgmentsReady}
					label={`${data.judgment_count} / ${data.candidate_count} candidates judged`}
				/>
			</CardContent>
			<CardFooter>
				<Button nativeButton={false} render={<Link to={nextAction.to} />}>
					{nextAction.label} <ArrowRight data-icon="inline-end" />
				</Button>
			</CardFooter>
		</Card>
	);
}

function SetupItem({ complete, label }: { complete: boolean; label: string }) {
	return (
		<div className="flex items-center gap-3 rounded-lg bg-muted/50 p-3 text-sm">
			<span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-background shadow-xs">
				{complete ? (
					<Check className="size-4" />
				) : (
					<span className="size-2 rounded-full bg-muted-foreground/40" />
				)}
			</span>
			<span className={cn(complete ? "font-medium" : "text-muted-foreground")}>{label}</span>
		</div>
	);
}

function RunSummary({
	title,
	description,
	run,
	empty,
	footer,
}: {
	title: string;
	description: string;
	run?: SearchEvalRun;
	empty: string;
	footer?: ReactNode;
}) {
	return (
		<Card className="min-w-0">
			<CardHeader>
				<CardTitle>{title}</CardTitle>
				<CardDescription>{description}</CardDescription>
				{run ? (
					<CardAction>
						<RunStatus run={run} />
					</CardAction>
				) : null}
			</CardHeader>
			<CardContent>
				{run ? (
					<div className="flex flex-col gap-5">
						<RunIdentity run={run} />
						{run.metrics ? <RunMetrics run={run} /> : null}
					</div>
				) : (
					<p className="text-sm text-pretty text-muted-foreground">{empty}</p>
				)}
			</CardContent>
			{footer ? <CardFooter>{footer}</CardFooter> : null}
		</Card>
	);
}

function ViewRunsButton() {
	return (
		<Button variant="outline" nativeButton={false} render={<Link to="/admin/evals/runs" />}>
			View runs <ArrowRight data-icon="inline-end" />
		</Button>
	);
}

function QueriesWorkspace({ data }: { data: OverviewData }) {
	const queryClient = useQueryClient();
	const activeQueries = data.queries.filter((query) => query.status === "active");
	const recipeIds = data.recipes.map((recipe) => recipe.id);
	const pool = useMutation({
		mutationFn: (queryId: number) => poolSearchEvalQuery({ data: { queryId, recipeIds } }),
		onSuccess: () => queryClient.invalidateQueries({ queryKey: OVERVIEW_QUERY_KEY }),
	});
	const refreshAll = useMutation({
		mutationFn: (queryIds: ReadonlyArray<number>) =>
			Promise.all(queryIds.map((queryId) => poolSearchEvalQuery({ data: { queryId, recipeIds } }))),
		onSuccess: () => queryClient.invalidateQueries({ queryKey: OVERVIEW_QUERY_KEY }),
	});
	const disable = useMutation({
		mutationFn: (queryId: number) => disableSearchEvalQuery({ data: { queryId } }),
		onSuccess: () => queryClient.invalidateQueries({ queryKey: OVERVIEW_QUERY_KEY }),
	});

	return (
		<div className="flex flex-col gap-5">
			<WorkspaceHeading
				title="Benchmark queries"
				description="Add real search needs, then generate or refresh candidates for each query."
			>
				<Button
					variant="outline"
					disabled={!activeQueries.length || refreshAll.isPending || pool.isPending}
					onClick={() => refreshAll.mutate(activeQueries.map((query) => query.id))}
				>
					{refreshAll.isPending ? (
						<Spinner data-icon="inline-start" />
					) : (
						<RefreshCw data-icon="inline-start" />
					)}
					Refresh all candidates
				</Button>
				<CreateQueryDialog />
			</WorkspaceHeading>

			{data.queries.length ? (
				<Card className="gap-0 overflow-hidden py-0">
					<CardHeader className="border-b py-5">
						<CardTitle>{activeQueries.length} active queries</CardTitle>
						<CardDescription>Candidate discovery is part of each query.</CardDescription>
					</CardHeader>
					<CardContent className="px-0">
						<div className="divide-y">
							{data.queries.map((query) => (
								<QueryRow
									key={query.id}
									query={query}
									pooling={pool.isPending && pool.variables === query.id}
									disabling={disable.isPending && disable.variables === query.id}
									onPool={() => pool.mutate(query.id)}
									onDisable={() => disable.mutate(query.id)}
								/>
							))}
						</div>
					</CardContent>
				</Card>
			) : (
				<QueryEmpty />
			)}

			{pool.error || refreshAll.error ? (
				<MutationError
					error={pool.error ?? refreshAll.error}
					fallback="Unable to generate candidates. Try again."
				/>
			) : null}
			{disable.error ? (
				<MutationError error={disable.error} fallback="Unable to disable the query. Try again." />
			) : null}
		</div>
	);
}

function WorkspaceHeading({
	title,
	description,
	children,
}: {
	title: string;
	description: string;
	children: ReactNode;
}) {
	return (
		<div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
			<div className="max-w-2xl">
				<h2 className="text-lg font-semibold tracking-tight">{title}</h2>
				<p className="text-sm text-pretty text-muted-foreground">{description}</p>
			</div>
			<div className="flex flex-col gap-2 min-[28rem]:flex-row">{children}</div>
		</div>
	);
}

function QueryEmpty() {
	return (
		<Empty className="min-h-72 border">
			<EmptyMedia variant="icon">
				<Database />
			</EmptyMedia>
			<EmptyHeader>
				<EmptyTitle>No benchmark queries yet</EmptyTitle>
				<EmptyDescription>
					Queries describe the search needs you want Mimeme to answer well.
				</EmptyDescription>
			</EmptyHeader>
			<EmptyContent>
				<CreateQueryDialog />
			</EmptyContent>
		</Empty>
	);
}

function QueryRow({
	query,
	pooling,
	disabling,
	onPool,
	onDisable,
}: {
	query: SearchEvalQuery;
	pooling: boolean;
	disabling: boolean;
	onPool: () => void;
	onDisable: () => void;
}) {
	const coverage = query.candidate_count
		? Math.round((query.judgment_count / query.candidate_count) * 100)
		: 0;
	return (
		<div className="@container/query p-4 sm:p-5">
			<div className="grid gap-4 @3xl/query:grid-cols-[minmax(0,1fr)_11rem_auto] @3xl/query:items-center">
				<div className="min-w-0">
					<div className="flex flex-wrap items-center gap-2">
						<p className="min-w-0 text-sm font-medium text-pretty">{query.text}</p>
						{query.status === "disabled" ? <Badge variant="secondary">Disabled</Badge> : null}
					</div>
					<div className="mt-2 flex flex-wrap gap-2">
						<Badge variant="outline">{query.intent}</Badge>
						<Badge variant="secondary">{query.source}</Badge>
					</div>
				</div>
				<div className="flex flex-col gap-2">
					<div className="flex items-center justify-between gap-3 text-xs">
						<span className="text-muted-foreground">Judged</span>
						<span className="tabular-nums">
							{query.judgment_count} / {query.candidate_count}
						</span>
					</div>
					<Progress value={coverage} aria-label={`${coverage}% judged for ${query.text}`} />
				</div>
				{query.status === "active" ? (
					<div className="flex flex-wrap gap-2 @3xl/query:justify-end">
						<Button variant="outline" size="sm" disabled={pooling} onClick={onPool}>
							{pooling ? (
								<Spinner data-icon="inline-start" />
							) : (
								<RefreshCw data-icon="inline-start" />
							)}
							{query.candidate_count ? "Refresh candidates" : "Generate candidates"}
						</Button>
						<Button
							size="sm"
							nativeButton={false}
							render={<Link to="/admin/evals/judge" search={{ query: query.id }} />}
						>
							Review <ArrowRight data-icon="inline-end" />
						</Button>
						<Button
							variant="ghost"
							size="icon-sm"
							aria-label={`Disable ${query.text}`}
							disabled={disabling}
							onClick={onDisable}
						>
							{disabling ? <Spinner /> : <Trash2 />}
						</Button>
					</div>
				) : null}
			</div>
		</div>
	);
}

function RunsWorkspace({ data }: { data: OverviewData }) {
	const queryClient = useQueryClient();
	const baseline = useMutation({
		mutationFn: (runId: string) => setSearchEvalBaseline({ data: { runId } }),
		onSuccess: () => queryClient.invalidateQueries({ queryKey: OVERVIEW_QUERY_KEY }),
	});
	const baselineRun = data.recent_runs.find((run) => run.id === data.baseline_run_id);
	const activeRun = data.recent_runs.find(
		(run) => run.status === "queued" || run.status === "running",
	);

	return (
		<div className="flex flex-col gap-5">
			<WorkspaceHeading
				title="Evaluation runs"
				description="Each run measures the currently served search system against a frozen benchmark revision."
			>
				<NewEvaluationDialog disabled={Boolean(activeRun)} recipes={data.recipes} />
			</WorkspaceHeading>

			{activeRun ? <RunProgress run={activeRun} /> : null}
			{baselineRun ? (
				<RunSummary
					title="Active baseline"
					description="New candidates are compared with this run."
					run={baselineRun}
					empty=""
				/>
			) : null}

			{data.recent_runs.length ? (
				<Card className="gap-0 overflow-hidden py-0">
					<CardHeader className="border-b py-5">
						<CardTitle>Recent runs</CardTitle>
						<CardDescription>Newest evaluation first.</CardDescription>
					</CardHeader>
					<CardContent className="px-0">
						<div className="divide-y">
							{data.recent_runs.map((run) => (
								<RunRow
									key={run.id}
									run={run}
									baselineRunId={data.baseline_run_id}
									settingBaseline={baseline.isPending && baseline.variables === run.id}
									onSetBaseline={() => baseline.mutate(run.id)}
								/>
							))}
						</div>
					</CardContent>
				</Card>
			) : (
				<RunsEmpty recipes={data.recipes} />
			)}

			{baseline.error ? (
				<MutationError error={baseline.error} fallback="Unable to set the baseline. Try again." />
			) : null}
		</div>
	);
}

function RunsEmpty({ recipes }: { recipes: ReadonlyArray<SearchEvalRecipe> }) {
	return (
		<Empty className="min-h-72 border">
			<EmptyMedia variant="icon">
				<Scale />
			</EmptyMedia>
			<EmptyHeader>
				<EmptyTitle>No evaluations yet</EmptyTitle>
				<EmptyDescription>
					Run the current search system against this benchmark to establish a baseline.
				</EmptyDescription>
			</EmptyHeader>
			<EmptyContent>
				<NewEvaluationDialog disabled={false} recipes={recipes} />
			</EmptyContent>
		</Empty>
	);
}

function RunRow({
	run,
	baselineRunId,
	settingBaseline,
	onSetBaseline,
}: {
	run: SearchEvalRun;
	baselineRunId: string | null;
	settingBaseline: boolean;
	onSetBaseline: () => void;
}) {
	const isBaseline = run.id === baselineRunId;
	return (
		<div className="@container/run p-4 sm:p-5">
			<div className="grid gap-4 @3xl/run:grid-cols-[minmax(0,1fr)_minmax(18rem,1fr)_auto] @3xl/run:items-center">
				<div className="min-w-0">
					<div className="flex flex-wrap items-center gap-2">
						<RunStatus run={run} />
						{isBaseline ? <Badge variant="outline">Baseline</Badge> : null}
					</div>
					<div className="mt-2">
						<RunIdentity run={run} />
					</div>
				</div>
				{run.metrics ? <RunMetrics run={run} layout="run-row" /> : <RunStateDetail run={run} />}
				<div className="flex flex-wrap gap-2 @3xl/run:justify-end">
					{run.status === "needs_judgments" ? (
						<Button size="sm" nativeButton={false} render={<Link to="/admin/evals/judge" />}>
							Review judgments
						</Button>
					) : null}
					{run.status === "complete" && !isBaseline ? (
						<Button variant="outline" size="sm" disabled={settingBaseline} onClick={onSetBaseline}>
							{settingBaseline ? <Spinner data-icon="inline-start" /> : null}
							Set as baseline
						</Button>
					) : null}
					{run.status === "complete" && baselineRunId && !isBaseline ? (
						<Button
							size="sm"
							nativeButton={false}
							render={
								<Link
									to="/admin/evals/compare"
									search={{ baseline: baselineRunId, candidate: run.id }}
								/>
							}
						>
							Compare <ArrowRight data-icon="inline-end" />
						</Button>
					) : null}
				</div>
			</div>
		</div>
	);
}

function RunStateDetail({ run }: { run: SearchEvalRun }) {
	if (run.status === "needs_judgments") {
		return (
			<p className="text-sm text-muted-foreground">
				<span className="font-medium text-foreground tabular-nums">{run.missing_judgments}</span>{" "}
				results need judgments before scoring.
			</p>
		);
	}
	if (run.error) return <p className="text-sm text-pretty text-destructive">{run.error}</p>;
	return <p className="text-sm text-muted-foreground capitalize">{run.phase ?? run.status}</p>;
}

function RunIdentity({ run }: { run: SearchEvalRun }) {
	return (
		<div className="min-w-0">
			<p className="font-medium">{run.recipe.label}</p>
			<p className="mt-1 truncate text-xs text-muted-foreground">
				Index {run.index_version ?? "unknown"} · Run {run.id.slice(0, 7)} · Snapshot{" "}
				{run.snapshot_id.slice(0, 7)}
			</p>
			<p className="mt-1 text-xs text-muted-foreground tabular-nums">
				{formatRunDate(run.created_at)}
			</p>
		</div>
	);
}

function RunStatus({ run }: { run: SearchEvalRun }) {
	const variant =
		run.status === "failed" ? "destructive" : run.status === "complete" ? "secondary" : "outline";
	return <Badge variant={variant}>{run.status.replace("_", " ")}</Badge>;
}

function RunMetrics({
	run,
	layout = "summary",
}: {
	run: SearchEvalRun;
	layout?: "summary" | "run-row";
}) {
	if (!run.metrics) return null;
	const metrics = [
		["nDCG@10", run.metrics.ndcg_at_10.toFixed(3)],
		["Success@5", `${(run.metrics.success_at_5 * 100).toFixed(1)}%`],
		["MRR@10", run.metrics.mrr_at_10.toFixed(3)],
		["p95", `${run.metrics.latency_p95_ms.toFixed(0)} ms`],
	] as const;
	return (
		<dl
			className={cn(
				layout === "run-row"
					? "grid grid-cols-2 gap-3 @xl/run:grid-cols-4"
					: "grid grid-cols-2 gap-4 sm:grid-cols-4",
			)}
		>
			{metrics.map(([label, value]) => (
				<div key={label} className="min-w-0">
					<dt className="text-xs text-muted-foreground">{label}</dt>
					<dd className="mt-1 font-semibold tabular-nums">{value}</dd>
				</div>
			))}
		</dl>
	);
}

function RunProgress({ run }: { run: SearchEvalRun }) {
	const progress = run.progress_total ? (run.progress_completed / run.progress_total) * 100 : 0;
	return (
		<Alert>
			<Spinner />
			<AlertTitle>Evaluation in progress</AlertTitle>
			<AlertDescription className="gap-3">
				<p className="capitalize">{run.phase ?? run.status}</p>
				<div className="flex w-full items-center gap-3">
					<Progress
						value={progress}
						aria-label={`${run.progress_completed} of ${run.progress_total} queries complete`}
					/>
					<span className="shrink-0 text-xs tabular-nums">
						{run.progress_completed}/{run.progress_total}
					</span>
				</div>
			</AlertDescription>
		</Alert>
	);
}

function NewEvaluationDialog({
	disabled,
	recipes,
}: {
	disabled: boolean;
	recipes: ReadonlyArray<SearchEvalRecipe>;
}) {
	const queryClient = useQueryClient();
	const [open, setOpen] = useState(false);
	const [recipeIds, setRecipeIds] = useState<ReadonlyArray<SearchEvalRecipeId>>(() =>
		recipes.map((recipe) => recipe.id),
	);
	const run = useMutation({
		mutationFn: () => createSearchEvalExperiment({ data: { recipeIds: [...recipeIds] } }),
		onSuccess: async () => {
			setOpen(false);
			await queryClient.invalidateQueries({ queryKey: OVERVIEW_QUERY_KEY });
		},
	});
	return (
		<Dialog open={open} onOpenChange={setOpen}>
			<DialogTrigger render={<Button disabled={disabled} aria-label="New evaluation" />}>
				<Play data-icon="inline-start" /> New evaluation
			</DialogTrigger>
			<DialogContent>
				<DialogHeader>
					<DialogTitle>New evaluation</DialogTitle>
					<DialogDescription>
						Evaluate the currently served search system against a frozen copy of the active queries
						and judgments.
					</DialogDescription>
				</DialogHeader>
				<FieldGroup>
					<Field>
						<FieldLabel id="evaluation-search-recipes">Recipes</FieldLabel>
						<ToggleGroup
							aria-labelledby="evaluation-search-recipes"
							variant="outline"
							value={[...recipeIds]}
							onValueChange={(values) => {
								const selected = new Set(values);
								setRecipeIds(
									recipes.flatMap((recipe) => (selected.has(recipe.id) ? [recipe.id] : [])),
								);
							}}
							className="grid w-full grid-cols-2 sm:grid-cols-5"
						>
							{recipes.map((recipe) => (
								<ToggleGroupItem key={recipe.id} value={recipe.id}>
									{recipe.label}
								</ToggleGroupItem>
							))}
						</ToggleGroup>
					</Field>
				</FieldGroup>
				{run.error ? (
					<MutationError error={run.error} fallback="Unable to start the evaluation. Try again." />
				) : null}
				<DialogFooter>
					<Button disabled={run.isPending || recipeIds.length === 0} onClick={() => run.mutate()}>
						{run.isPending ? (
							<Spinner data-icon="inline-start" />
						) : (
							<Play data-icon="inline-start" />
						)}
						Run {recipeIds.length} {recipeIds.length === 1 ? "recipe" : "recipes"}
					</Button>
				</DialogFooter>
			</DialogContent>
		</Dialog>
	);
}

function CreateQueryDialog() {
	const queryClient = useQueryClient();
	const [open, setOpen] = useState(false);
	const [text, setText] = useState("");
	const [intent, setIntent] = useState<SearchEvalIntent>("reaction");
	const [source, setSource] = useState<SearchEvalQuerySource>("human");
	const create = useMutation({
		mutationFn: () => createSearchEvalQuery({ data: { text, intent, source } }),
		onSuccess: async () => {
			setText("");
			setOpen(false);
			await queryClient.invalidateQueries({ queryKey: OVERVIEW_QUERY_KEY });
		},
	});
	return (
		<Dialog open={open} onOpenChange={setOpen}>
			<DialogTrigger render={<Button size="sm" aria-label="Add query" />}>
				<CirclePlus data-icon="inline-start" /> Add query
			</DialogTrigger>
			<DialogContent>
				<form
					className="flex flex-col gap-5"
					onSubmit={(event) => {
						event.preventDefault();
						create.mutate();
					}}
				>
					<DialogHeader>
						<DialogTitle>Add query</DialogTitle>
						<DialogDescription>Add a phrase that represents a real search need.</DialogDescription>
					</DialogHeader>
					<FieldGroup>
						<Field>
							<FieldLabel htmlFor="eval-query">Query</FieldLabel>
							<Input
								id="eval-query"
								value={text}
								onChange={(event) => setText(event.target.value)}
								placeholder="me pretending everything is okay"
							/>
						</Field>
						<div className="grid gap-4 sm:grid-cols-2">
							<QueryIntentSelect value={intent} onChange={setIntent} />
							<QuerySourceSelect value={source} onChange={setSource} />
						</div>
					</FieldGroup>
					{create.error ? (
						<MutationError error={create.error} fallback="Unable to add the query. Try again." />
					) : null}
					<DialogFooter>
						<Button type="submit" disabled={!text.trim() || create.isPending}>
							{create.isPending ? <Spinner data-icon="inline-start" /> : null}
							Add query
						</Button>
					</DialogFooter>
				</form>
			</DialogContent>
		</Dialog>
	);
}

function QueryIntentSelect({
	value,
	onChange,
}: {
	value: SearchEvalIntent;
	onChange: (value: SearchEvalIntent) => void;
}) {
	return (
		<Field>
			<FieldLabel htmlFor="eval-intent">Intent</FieldLabel>
			<Select
				value={value}
				onValueChange={(nextValue) => {
					const next = SEARCH_EVAL_INTENTS.find((item) => item === nextValue);
					if (next) onChange(next);
				}}
			>
				<SelectTrigger id="eval-intent" className="w-full">
					<SelectValue />
				</SelectTrigger>
				<SelectContent>
					<SelectGroup>
						{SEARCH_EVAL_INTENTS.map((intent) => (
							<SelectItem key={intent} value={intent}>
								{intent}
							</SelectItem>
						))}
					</SelectGroup>
				</SelectContent>
			</Select>
		</Field>
	);
}

function QuerySourceSelect({
	value,
	onChange,
}: {
	value: SearchEvalQuerySource;
	onChange: (value: SearchEvalQuerySource) => void;
}) {
	return (
		<Field>
			<FieldLabel htmlFor="eval-source">Source</FieldLabel>
			<Select
				value={value}
				onValueChange={(nextValue) => {
					if (nextValue === "human" || nextValue === "production" || nextValue === "synthetic") {
						onChange(nextValue);
					}
				}}
			>
				<SelectTrigger id="eval-source" className="w-full">
					<SelectValue />
				</SelectTrigger>
				<SelectContent>
					<SelectGroup>
						<SelectItem value="human">Human</SelectItem>
						<SelectItem value="production">Production</SelectItem>
						<SelectItem value="synthetic">Synthetic</SelectItem>
					</SelectGroup>
				</SelectContent>
			</Select>
		</Field>
	);
}

function MutationError({ error, fallback }: { error: unknown; fallback: string }) {
	return (
		<Alert variant="destructive">
			<AlertTriangle />
			<AlertTitle>Action failed</AlertTitle>
			<AlertDescription>{adminErrorMessage(error, fallback)}</AlertDescription>
		</Alert>
	);
}

function formatRunDate(value: string): string {
	return RUN_DATE_FORMATTER.format(new Date(value));
}

function OverviewSkeleton() {
	return (
		<div className="flex flex-col gap-6">
			<div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
				{Array.from({ length: 4 }, (_, index) => (
					<Skeleton key={index} className="h-32" />
				))}
			</div>
			<Skeleton className="h-64" />
			<div className="grid gap-4 lg:grid-cols-2">
				<Skeleton className="h-72" />
				<Skeleton className="h-72" />
			</div>
		</div>
	);
}

function ListSkeleton() {
	return (
		<div className="flex flex-col gap-5">
			<Skeleton className="h-16" />
			<Skeleton className="h-[32rem]" />
		</div>
	);
}
