import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
	DialogTrigger,
} from "@/components/ui/dialog";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
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
	createSearchEvalRun,
	disableSearchEvalQuery,
	poolSearchEvalQuery,
	SEARCH_EVAL_INTENTS,
	searchEvalOverviewQueryOptions,
	setSearchEvalBaseline,
	type SearchEvalIntent,
	type SearchEvalOverview,
	type SearchEvalQuerySource,
	type SearchEvalRun,
	type SearchEvalRunMode,
} from "@/lib/admin/search-eval-api";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import {
	AlertTriangle,
	ArrowLeft,
	ArrowRight,
	Check,
	CirclePlus,
	Play,
	RefreshCw,
	Trash2,
} from "lucide-react";
import { createContext, use, useMemo, useState, type ReactNode } from "react";

type WizardStep = 1 | 2 | 3 | 4 | 5;

const STEPS: ReadonlyArray<{ readonly id: WizardStep; readonly label: string }> = [
	{ id: 1, label: "Queries" },
	{ id: 2, label: "Pool" },
	{ id: 3, label: "Judge" },
	{ id: 4, label: "Run" },
	{ id: 5, label: "Compare" },
];

interface WizardContextValue {
	state: {
		readonly step: WizardStep;
		readonly data: SearchEvalOverview;
		readonly furthestStep: WizardStep;
	};
	actions: {
		readonly goTo: (step: WizardStep) => void;
	};
	meta: {
		readonly steps: typeof STEPS;
	};
}

const WizardContext = createContext<WizardContextValue | null>(null);

function useWizard(): WizardContextValue {
	const wizard = use(WizardContext);
	if (!wizard) throw new Error("Evaluation wizard components require WizardProvider");
	return wizard;
}

/** Guided creation flow for the shared search benchmark. */
export function SearchEvalOverview() {
	const overview = useQuery(searchEvalOverviewQueryOptions());
	if (overview.isPending) return <WizardSkeleton />;
	if (!overview.data) return null;
	return (
		<WizardProvider data={overview.data}>
			<SearchEvalWizard />
		</WizardProvider>
	);
}

function WizardProvider({ data, children }: { data: SearchEvalOverview; children: ReactNode }) {
	const [step, setStep] = useState<WizardStep>(() => nextRequiredStep(data));
	const furthestStep = nextRequiredStep(data);
	const value = useMemo<WizardContextValue>(
		() => ({
			state: { step, data, furthestStep },
			actions: { goTo: setStep },
			meta: { steps: STEPS },
		}),
		[data, furthestStep, step],
	);
	return <WizardContext value={value}>{children}</WizardContext>;
}

function SearchEvalWizard() {
	return (
		<div className="mx-auto flex w-full max-w-4xl flex-col gap-6">
			<WizardProgress />
			<WizardStepContent />
		</div>
	);
}

function WizardStepContent() {
	const { state } = useWizard();
	switch (state.step) {
		case 1:
			return <QueriesStep />;
		case 2:
			return <PoolStep />;
		case 3:
			return <JudgeStep />;
		case 4:
			return <RunStep />;
		case 5:
			return <CompareStep />;
	}
}

function WizardProgress() {
	const { state, actions, meta } = useWizard();
	return (
		<ol className="grid grid-cols-5 border" aria-label="Evaluation steps">
			{meta.steps.map((item) => {
				const complete = item.id < state.furthestStep;
				const disabled = item.id > state.furthestStep;
				return (
					<li key={item.id} className="border-l first:border-l-0">
						<button
							type="button"
							disabled={disabled}
							aria-current={state.step === item.id ? "step" : undefined}
							className="flex min-h-16 w-full flex-col items-center justify-center gap-1 px-2 text-xs hover:bg-accent disabled:cursor-not-allowed disabled:opacity-40 data-[current=true]:bg-accent sm:flex-row sm:text-sm"
							data-current={state.step === item.id}
							onClick={() => actions.goTo(item.id)}
						>
							<span className="flex size-5 items-center justify-center border text-[10px]">
								{complete ? <Check className="size-3" /> : item.id}
							</span>
							<span>{item.label}</span>
						</button>
					</li>
				);
			})}
		</ol>
	);
}

function QueriesStep() {
	const { state } = useWizard();
	const queryClient = useQueryClient();
	const disable = useMutation({
		mutationFn: (queryId: number) => disableSearchEvalQuery({ data: { queryId } }),
		onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "search-evals"] }),
	});
	const activeQueries = state.data.queries.filter((query) => query.status === "active");
	return (
		<WizardPanel
			title="Add search queries"
			description="Use phrases people would actually search for."
		>
			<div className="flex justify-end">
				<CreateQueryDialog />
			</div>
			{activeQueries.length ? (
				<div className="divide-y border">
					{activeQueries.map((query) => (
						<div key={query.id} className="flex items-center gap-4 p-4">
							<div className="min-w-0 flex-1">
								<p className="text-sm font-medium">{query.text}</p>
								<div className="mt-2 flex gap-2">
									<Badge variant="outline">{query.intent}</Badge>
									<Badge variant="secondary">{query.source}</Badge>
								</div>
							</div>
							<Button
								variant="ghost"
								size="icon-sm"
								aria-label={`Disable ${query.text}`}
								disabled={disable.isPending}
								onClick={() => disable.mutate(query.id)}
							>
								<Trash2 />
							</Button>
						</div>
					))}
				</div>
			) : (
				<Empty className="border">
					<EmptyHeader>
						<EmptyTitle>No queries yet</EmptyTitle>
						<EmptyDescription>Add at least one query.</EmptyDescription>
					</EmptyHeader>
				</Empty>
			)}
			{disable.error ? (
				<MutationError error={disable.error} fallback="Could not disable the query." />
			) : null}
			<WizardActions>
				<WizardNext step={2} disabled={!activeQueries.length}>
					Continue to pool
				</WizardNext>
			</WizardActions>
		</WizardPanel>
	);
}

function PoolStep() {
	const { state } = useWizard();
	const queryClient = useQueryClient();
	const pool = useMutation({
		mutationFn: (queryId: number) => poolSearchEvalQuery({ data: { queryId } }),
		onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "search-evals"] }),
	});
	const activeQueries = state.data.queries.filter((query) => query.status === "active");
	const complete =
		activeQueries.length > 0 && activeQueries.every((query) => query.candidate_count > 0);
	return (
		<WizardPanel
			title="Build candidate pools"
			description="Collect results from image and hybrid search."
		>
			<div className="divide-y border">
				{activeQueries.map((query) => (
					<div key={query.id} className="flex items-center gap-4 p-4">
						<div className="min-w-0 flex-1">
							<p className="truncate text-sm font-medium">{query.text}</p>
							<p className="mt-1 text-xs text-muted-foreground">
								{query.candidate_count ? `${query.candidate_count} candidates` : "Not pooled"}
							</p>
						</div>
						<Button
							variant={query.candidate_count ? "outline" : "default"}
							size="sm"
							disabled={pool.isPending}
							onClick={() => pool.mutate(query.id)}
						>
							{pool.isPending && pool.variables === query.id ? (
								<Spinner data-icon="inline-start" />
							) : (
								<RefreshCw data-icon="inline-start" />
							)}
							{query.candidate_count ? "Refresh pool" : "Build pool"}
						</Button>
					</div>
				))}
			</div>
			{pool.error ? (
				<MutationError error={pool.error} fallback="Could not build the pool." />
			) : null}
			<WizardActions>
				<WizardBack step={1}>Queries</WizardBack>
				<WizardNext step={3} disabled={!complete}>
					Continue to judge
				</WizardNext>
			</WizardActions>
		</WizardPanel>
	);
}

function JudgeStep() {
	const { state } = useWizard();
	const activeQueries = state.data.queries.filter((query) => query.status === "active");
	const complete =
		activeQueries.length > 0 &&
		state.data.unjudged_count === 0 &&
		activeQueries.every((query) => query.relevant_count > 0);
	const progress = state.data.candidate_count
		? (state.data.judgment_count / state.data.candidate_count) * 100
		: 0;
	return (
		<WizardPanel title="Judge relevance" description="Grade every candidate from 0 to 3.">
			<div className="flex items-center gap-3">
				<Progress
					value={progress}
					aria-label={`${state.data.judgment_count} of ${state.data.candidate_count} graded`}
				/>
				<span className="shrink-0 text-sm tabular-nums">
					{state.data.judgment_count}/{state.data.candidate_count}
				</span>
			</div>
			<div className="divide-y border">
				{activeQueries.map((query) => {
					const queryComplete =
						query.candidate_count > 0 &&
						query.judgment_count === query.candidate_count &&
						query.relevant_count > 0;
					return (
						<div key={query.id} className="flex items-center gap-4 p-4">
							<div className="min-w-0 flex-1">
								<p className="truncate text-sm font-medium">{query.text}</p>
								<p className="mt-1 text-xs text-muted-foreground">
									{query.judgment_count}/{query.candidate_count} graded · {query.relevant_count}{" "}
									relevant
								</p>
							</div>
							{queryComplete ? (
								<Badge variant="outline">
									<Check /> Complete
								</Badge>
							) : null}
							<Button
								size="sm"
								nativeButton={false}
								render={<Link to="/admin/evals/judge" search={{ query: query.id }} />}
							>
								Judge <ArrowRight data-icon="inline-end" />
							</Button>
						</div>
					);
				})}
			</div>
			<WizardActions>
				<WizardBack step={2}>Pool</WizardBack>
				<WizardNext step={4} disabled={!complete}>
					Continue to run
				</WizardNext>
			</WizardActions>
		</WizardPanel>
	);
}

function RunStep() {
	const { state } = useWizard();
	const queryClient = useQueryClient();
	const [mode, setMode] = useState<SearchEvalRunMode>("hybrid");
	const run = useMutation({
		mutationFn: () => createSearchEvalRun({ data: { mode } }),
		onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "search-evals"] }),
	});
	const activeRun = state.data.recent_runs.find(
		(item) => item.status === "queued" || item.status === "running",
	);
	const completeRuns = state.data.recent_runs.filter((item) => item.status === "complete");
	const needsJudgments = state.data.recent_runs.find((item) => item.status === "needs_judgments");
	return (
		<WizardPanel title="Run the evaluation" description="Choose one search mode.">
			<ToggleGroup
				variant="outline"
				value={[mode]}
				onValueChange={([value]) => {
					if (value === "image" || value === "hybrid") setMode(value);
				}}
				className="w-full"
			>
				<ToggleGroupItem value="image" className="flex-1">
					Image
				</ToggleGroupItem>
				<ToggleGroupItem value="hybrid" className="flex-1">
					Hybrid
				</ToggleGroupItem>
			</ToggleGroup>
			<Button disabled={Boolean(activeRun) || run.isPending} onClick={() => run.mutate()}>
				{run.isPending ? <Spinner data-icon="inline-start" /> : <Play data-icon="inline-start" />}
				Run {mode} search
			</Button>
			{activeRun ? (
				<RunProgress
					run={activeRun}
					onRefresh={() => queryClient.invalidateQueries({ queryKey: ["admin", "search-evals"] })}
				/>
			) : null}
			{needsJudgments ? (
				<Alert>
					<AlertTitle>{needsJudgments.missing_judgments} new results need grades</AlertTitle>
					<AlertDescription>
						<Button
							className="mt-3"
							size="sm"
							nativeButton={false}
							render={<Link to="/admin/evals/judge" />}
						>
							Review results
						</Button>
					</AlertDescription>
				</Alert>
			) : null}
			{completeRuns.length ? (
				<CompletedRuns runs={completeRuns} baselineRunId={state.data.baseline_run_id} />
			) : null}
			{run.error ? (
				<MutationError error={run.error} fallback="The evaluation could not start." />
			) : null}
			<WizardActions>
				<WizardBack step={3}>Judge</WizardBack>
				<WizardNext step={5} disabled={!completeRuns.length}>
					Continue to compare
				</WizardNext>
			</WizardActions>
		</WizardPanel>
	);
}

function CompareStep() {
	const { state } = useWizard();
	const queryClient = useQueryClient();
	const baseline = useMutation({
		mutationFn: (runId: string) => setSearchEvalBaseline({ data: { runId } }),
		onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "search-evals"] }),
	});
	const completeRuns = state.data.recent_runs.filter((run) => run.status === "complete");
	const candidates = completeRuns.filter((run) => run.id !== state.data.baseline_run_id);
	return (
		<WizardPanel
			title="Compare with the baseline"
			description="Choose the run you want to evaluate."
		>
			{!state.data.baseline_run_id ? (
				<div className="divide-y border">
					{completeRuns.map((run) => (
						<div key={run.id} className="flex items-center justify-between gap-4 p-4">
							<RunLabel run={run} />
							<Button
								size="sm"
								disabled={baseline.isPending}
								onClick={() => baseline.mutate(run.id)}
							>
								Set baseline
							</Button>
						</div>
					))}
				</div>
			) : candidates.length ? (
				<div className="divide-y border">
					{candidates.map((run) => (
						<div key={run.id} className="flex items-center justify-between gap-4 p-4">
							<RunLabel run={run} />
							<Button
								nativeButton={false}
								render={
									<Link
										to="/admin/evals/compare"
										search={{
											baseline: state.data.baseline_run_id ?? undefined,
											candidate: run.id,
										}}
									/>
								}
							>
								Compare <ArrowRight data-icon="inline-end" />
							</Button>
						</div>
					))}
				</div>
			) : (
				<Empty className="border">
					<EmptyHeader>
						<EmptyTitle>No candidate run yet</EmptyTitle>
						<EmptyDescription>Run another mode or index version first.</EmptyDescription>
					</EmptyHeader>
				</Empty>
			)}
			{baseline.error ? (
				<MutationError error={baseline.error} fallback="Could not set the baseline." />
			) : null}
			<WizardActions>
				<WizardBack step={4}>Run</WizardBack>
			</WizardActions>
		</WizardPanel>
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
		onSuccess: () => {
			setText("");
			setOpen(false);
			return queryClient.invalidateQueries({ queryKey: ["admin", "search-evals"] });
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
						<DialogDescription>Add a phrase to the evaluation.</DialogDescription>
					</DialogHeader>
					<FieldGroup>
						<Field>
							<FieldLabel htmlFor="eval-query">Query</FieldLabel>
							<Input
								id="eval-query"
								value={text}
								onChange={(event) => setText(event.target.value)}
							/>
						</Field>
						<div className="grid gap-4 sm:grid-cols-2">
							<Field>
								<FieldLabel htmlFor="eval-intent">Intent</FieldLabel>
								<Select
									value={intent}
									onValueChange={(value) => {
										const next = SEARCH_EVAL_INTENTS.find((item) => item === value);
										if (next) setIntent(next);
									}}
								>
									<SelectTrigger id="eval-intent" className="w-full">
										<SelectValue />
									</SelectTrigger>
									<SelectContent>
										<SelectGroup>
											{SEARCH_EVAL_INTENTS.map((value) => (
												<SelectItem key={value} value={value}>
													{value}
												</SelectItem>
											))}
										</SelectGroup>
									</SelectContent>
								</Select>
							</Field>
							<Field>
								<FieldLabel htmlFor="eval-source">Source</FieldLabel>
								<Select
									value={source}
									onValueChange={(value) => {
										if (value === "human" || value === "production" || value === "synthetic")
											setSource(value);
									}}
								>
									<SelectTrigger id="eval-source" className="w-full">
										<SelectValue />
									</SelectTrigger>
									<SelectContent>
										<SelectGroup>
											<SelectItem value="human">human</SelectItem>
											<SelectItem value="production">production</SelectItem>
											<SelectItem value="synthetic">synthetic</SelectItem>
										</SelectGroup>
									</SelectContent>
								</Select>
							</Field>
						</div>
					</FieldGroup>
					{create.error ? (
						<MutationError error={create.error} fallback="Could not add the query." />
					) : null}
					<DialogFooter>
						<Button type="submit" disabled={!text.trim() || create.isPending}>
							{create.isPending ? <Spinner data-icon="inline-start" /> : null}Add query
						</Button>
					</DialogFooter>
				</form>
			</DialogContent>
		</Dialog>
	);
}

function WizardPanel({
	title,
	description,
	children,
}: {
	title: string;
	description: string;
	children: ReactNode;
}) {
	return (
		<Card>
			<CardHeader>
				<CardTitle>{title}</CardTitle>
				<CardDescription>{description}</CardDescription>
			</CardHeader>
			<CardContent className="flex flex-col gap-5">{children}</CardContent>
		</Card>
	);
}

function WizardActions({ children }: { children: ReactNode }) {
	return <div className="flex items-center justify-between border-t pt-5">{children}</div>;
}

function WizardBack({ step, children }: { step: WizardStep; children: ReactNode }) {
	const { actions } = useWizard();
	return (
		<Button variant="ghost" onClick={() => actions.goTo(step)}>
			<ArrowLeft data-icon="inline-start" />
			{children}
		</Button>
	);
}

function WizardNext({
	step,
	disabled,
	children,
}: {
	step: WizardStep;
	disabled: boolean;
	children: ReactNode;
}) {
	const { actions } = useWizard();
	return (
		<Button disabled={disabled} onClick={() => actions.goTo(step)}>
			{children}
			<ArrowRight data-icon="inline-end" />
		</Button>
	);
}

function RunProgress({ run, onRefresh }: { run: SearchEvalRun; onRefresh: () => void }) {
	const progress = run.progress_total ? (run.progress_completed / run.progress_total) * 100 : 0;
	return (
		<div className="flex flex-col gap-3 border p-4">
			<div className="flex items-center justify-between gap-4">
				<span className="text-sm font-medium">{run.phase ?? run.status}</span>
				<Button variant="ghost" size="sm" onClick={onRefresh}>
					<RefreshCw data-icon="inline-start" />
					Refresh
				</Button>
			</div>
			<Progress
				value={progress}
				aria-label={`${run.progress_completed} of ${run.progress_total} queries`}
			/>
		</div>
	);
}

function CompletedRuns({
	runs,
	baselineRunId,
}: {
	runs: SearchEvalRun[];
	baselineRunId: string | null;
}) {
	return (
		<div className="divide-y border">
			{runs.slice(0, 4).map((run) => (
				<div key={run.id} className="flex items-center justify-between gap-4 p-4">
					<RunLabel run={run} />
					{run.id === baselineRunId ? (
						<Badge variant="outline">Baseline</Badge>
					) : (
						<Badge variant="secondary">Complete</Badge>
					)}
				</div>
			))}
		</div>
	);
}

function RunLabel({ run }: { run: SearchEvalRun }) {
	return (
		<div className="min-w-0">
			<div className="flex items-center gap-2">
				<span className="font-medium capitalize">{run.mode}</span>
				{run.metrics ? (
					<span className="text-sm tabular-nums">nDCG {run.metrics.ndcg_at_10.toFixed(3)}</span>
				) : null}
			</div>
			<p className="mt-1 truncate text-xs text-muted-foreground">
				{run.index_version ?? "No index version"}
			</p>
		</div>
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

function nextRequiredStep(data: SearchEvalOverview): WizardStep {
	const activeQueries = data.queries.filter((query) => query.status === "active");
	if (!activeQueries.length) return 1;
	if (activeQueries.some((query) => query.candidate_count === 0)) return 2;
	if (data.unjudged_count > 0 || activeQueries.some((query) => query.relevant_count === 0))
		return 3;
	const completeRuns = data.recent_runs.filter((run) => run.status === "complete");
	if (!data.baseline_run_id || !completeRuns.some((run) => run.id !== data.baseline_run_id))
		return 4;
	return 5;
}

function WizardSkeleton() {
	return (
		<div className="mx-auto flex w-full max-w-4xl flex-col gap-6">
			<Skeleton className="h-16" />
			<Skeleton className="h-[32rem]" />
		</div>
	);
}
