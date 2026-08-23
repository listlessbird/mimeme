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
import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from "@/components/ui/table";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { adminErrorMessage } from "@/lib/admin/api";
import {
	createSearchEvalQuery,
	createSearchEvalRun,
	disableSearchEvalQuery,
	finalizeSearchEvalRun,
	poolSearchEvalQuery,
	SEARCH_EVAL_INTENTS,
	searchEvalOverviewQueryOptions,
	setSearchEvalBaseline,
	type SearchEvalIntent,
	type SearchEvalQuerySource,
	type SearchEvalRun,
	type SearchEvalRunMode,
} from "@/lib/admin/search-eval-api";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import {
	AlertTriangle,
	ArrowRight,
	Baseline,
	CirclePlus,
	Play,
	RefreshCw,
	Trash2,
} from "lucide-react";
import { useState } from "react";

/** Core Search query, judgment, run, and baseline workspace. */
export function SearchEvalOverview() {
	const overview = useQuery(searchEvalOverviewQueryOptions());
	if (overview.isPending) return <OverviewSkeleton />;
	if (!overview.data) return null;
	return (
		<div className="flex flex-col gap-8">
			<WorkspaceStatus data={overview.data} />
			<QueryTable data={overview.data} />
			<RunTable data={overview.data} />
		</div>
	);
}

function WorkspaceStatus({ data }: { data: NonNullable<ReturnType<typeof useOverviewData>> }) {
	const judgedPercent = data.candidate_count
		? (data.judgment_count / data.candidate_count) * 100
		: 0;
	const runAvailability = data.recent_runs.some(
		(run) => run.status === "queued" || run.status === "running",
	)
		? "busy"
		: data.active_query_count > 0 && data.unjudged_count === 0
			? "ready"
			: "incomplete";
	return (
		<section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(18rem,0.55fr)]">
			<Card>
				<CardHeader>
					<CardTitle>Judgment coverage</CardTitle>
					<CardDescription>
						{data.judgment_count} of {data.candidate_count} pooled results graded
					</CardDescription>
					<CardAction>
						<Badge variant={data.unjudged_count ? "secondary" : "outline"}>
							{data.unjudged_count} left
						</Badge>
					</CardAction>
				</CardHeader>
				<CardContent className="flex flex-col gap-4">
					<Progress value={judgedPercent} aria-label={`${judgedPercent.toFixed(0)}% judged`} />
					<div className="flex flex-wrap gap-x-8 gap-y-2 text-sm">
						<Measurement value={data.active_query_count} label="active queries" />
						<Measurement value={data.candidate_count} label="pooled memes" />
						<Measurement value={data.baseline_run_id ? "set" : "none"} label="baseline" />
					</div>
				</CardContent>
			</Card>
			<RunControl availability={runAvailability} />
		</section>
	);
}

function Measurement({ value, label }: { value: number | string; label: string }) {
	return (
		<div className="flex items-baseline gap-2">
			<span className="text-lg font-semibold tabular-nums">{value}</span>
			<span className="text-xs text-muted-foreground">{label}</span>
		</div>
	);
}

type RunAvailability = "ready" | "busy" | "incomplete";

function RunControl({ availability }: { availability: RunAvailability }) {
	const queryClient = useQueryClient();
	const [mode, setMode] = useState<SearchEvalRunMode>("hybrid");
	const run = useMutation({
		mutationFn: () => createSearchEvalRun({ data: { mode } }),
		onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "search-evals"] }),
	});
	return (
		<Card>
			<CardHeader>
				<CardTitle>Run the benchmark</CardTitle>
				<CardDescription>The run stays pinned to one active index version.</CardDescription>
			</CardHeader>
			<CardContent className="flex flex-col gap-4">
				<ToggleGroup
					variant="outline"
					value={[mode]}
					onValueChange={([value]) => {
						if (value === "image" || value === "hybrid") setMode(value);
					}}
					className="w-full"
				>
					<ToggleGroupItem value="image" className="flex-1">
						image
					</ToggleGroupItem>
					<ToggleGroupItem value="hybrid" className="flex-1">
						hybrid
					</ToggleGroupItem>
				</ToggleGroup>
				<Button disabled={availability !== "ready" || run.isPending} onClick={() => run.mutate()}>
					{run.isPending ? <Spinner data-icon="inline-start" /> : <Play data-icon="inline-start" />}
					{run.isPending ? "Queueing run" : "Run eval"}
				</Button>
				{availability === "incomplete" ? (
					<p className="text-xs text-muted-foreground">
						Pool and judge every active query before starting a run.
					</p>
				) : null}
				{availability === "busy" ? (
					<p className="text-xs text-muted-foreground">
						One evaluation is already using the search worker.
					</p>
				) : null}
				{run.error ? (
					<p className="text-xs text-destructive">
						{adminErrorMessage(run.error, "The eval run failed.")}
					</p>
				) : null}
			</CardContent>
		</Card>
	);
}

function QueryTable({ data }: { data: NonNullable<ReturnType<typeof useOverviewData>> }) {
	const queryClient = useQueryClient();
	const pool = useMutation({
		mutationFn: (queryId: number) => poolSearchEvalQuery({ data: { queryId } }),
		onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "search-evals"] }),
	});
	const disable = useMutation({
		mutationFn: (queryId: number) => disableSearchEvalQuery({ data: { queryId } }),
		onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "search-evals"] }),
	});
	return (
		<section className="flex flex-col gap-3">
			<div className="flex items-end justify-between gap-4">
				<div className="flex flex-col gap-1">
					<h2 className="font-semibold">Queries</h2>
					<p className="text-sm text-muted-foreground">
						Write the need first. The pool comes from current image and hybrid search.
					</p>
				</div>
				<CreateQueryDialog />
			</div>
			{data.queries.length === 0 ? (
				<Empty className="border">
					<EmptyHeader>
						<EmptyTitle>No search queries yet</EmptyTitle>
						<EmptyDescription>Add the first query without choosing a target meme.</EmptyDescription>
					</EmptyHeader>
				</Empty>
			) : (
				<div className="border">
					<Table>
						<TableHeader>
							<TableRow>
								<TableHead>Query</TableHead>
								<TableHead>Intent</TableHead>
								<TableHead>Judged</TableHead>
								<TableHead className="text-right">Actions</TableHead>
							</TableRow>
						</TableHeader>
						<TableBody>
							{data.queries.map((query) => (
								<TableRow
									key={query.id}
									data-state={query.status === "disabled" ? "selected" : undefined}
								>
									<TableCell className="max-w-[34rem] whitespace-normal">
										<span
											className={
												query.status === "disabled" ? "text-muted-foreground line-through" : ""
											}
										>
											{query.text}
										</span>
									</TableCell>
									<TableCell>
										<Badge variant="outline">{query.intent}</Badge>
									</TableCell>
									<TableCell className="tabular-nums">
										{query.judgment_count}/{query.candidate_count}
									</TableCell>
									<TableCell>
										<div className="flex justify-end gap-1">
											{query.status === "active" ? (
												<>
													<Button
														variant="ghost"
														size="sm"
														disabled={pool.isPending}
														onClick={() => pool.mutate(query.id)}
													>
														<RefreshCw data-icon="inline-start" />
														Pool
													</Button>
													<Button
														variant="ghost"
														size="sm"
														nativeButton={false}
														render={<Link to="/admin/evals/judge" search={{ query: query.id }} />}
													>
														Judge <ArrowRight data-icon="inline-end" />
													</Button>
													<Button
														variant="ghost"
														size="icon-sm"
														aria-label={`Disable ${query.text}`}
														disabled={disable.isPending}
														onClick={() => disable.mutate(query.id)}
													>
														<Trash2 />
													</Button>
												</>
											) : null}
										</div>
									</TableCell>
								</TableRow>
							))}
						</TableBody>
					</Table>
				</div>
			)}
			{pool.error || disable.error ? (
				<Alert variant="destructive">
					<AlertTriangle />
					<AlertTitle>Query action failed</AlertTitle>
					<AlertDescription>
						{adminErrorMessage(pool.error ?? disable.error, "Try the action again.")}
					</AlertDescription>
				</Alert>
			) : null}
		</section>
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
						<DialogTitle>Add a search need</DialogTitle>
						<DialogDescription>
							Write what someone would type without choosing the answer first.
						</DialogDescription>
					</DialogHeader>
					<FieldGroup>
						<Field>
							<FieldLabel htmlFor="eval-query">Query</FieldLabel>
							<Input
								id="eval-query"
								value={text}
								onChange={(event) => setText(event.target.value)}
								placeholder="when the deploy works locally but production is broken"
							/>
						</Field>
						<div className="grid gap-4 sm:grid-cols-2">
							<Field>
								<FieldLabel htmlFor="eval-intent">Intent</FieldLabel>
								<Select
									value={intent}
									onValueChange={(value) => {
										const next = SEARCH_EVAL_INTENTS.find((intent) => intent === value);
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
										if (value === "human" || value === "production" || value === "synthetic") {
											setSource(value);
										}
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
						<p className="text-sm text-destructive">
							{adminErrorMessage(create.error, "Could not add the query.")}
						</p>
					) : null}
					<DialogFooter>
						<Button type="submit" disabled={!text.trim() || create.isPending}>
							{create.isPending ? <Spinner data-icon="inline-start" /> : null}
							Add and pool later
						</Button>
					</DialogFooter>
				</form>
			</DialogContent>
		</Dialog>
	);
}

function RunTable({ data }: { data: NonNullable<ReturnType<typeof useOverviewData>> }) {
	const queryClient = useQueryClient();
	const finalize = useMutation({
		mutationFn: (runId: string) => finalizeSearchEvalRun({ data: { runId } }),
		onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "search-evals"] }),
	});
	const baseline = useMutation({
		mutationFn: (runId: string) => setSearchEvalBaseline({ data: { runId } }),
		onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "search-evals"] }),
	});
	return (
		<section className="flex flex-col gap-3">
			<div className="flex flex-col gap-1">
				<h2 className="font-semibold">Runs</h2>
				<p className="text-sm text-muted-foreground">
					Raw rankings stay available for later comparisons.
				</p>
			</div>
			{data.recent_runs.length === 0 ? (
				<Empty className="border">
					<EmptyHeader>
						<EmptyTitle>No runs yet</EmptyTitle>
						<EmptyDescription>
							Finish the judgment queue, then run image or hybrid search.
						</EmptyDescription>
					</EmptyHeader>
				</Empty>
			) : (
				<div className="border">
					<Table>
						<TableHeader>
							<TableRow>
								<TableHead>Run</TableHead>
								<TableHead>Status</TableHead>
								<TableHead>nDCG@10</TableHead>
								<TableHead>Judged@10</TableHead>
								<TableHead>p95</TableHead>
								<TableHead className="text-right">Actions</TableHead>
							</TableRow>
						</TableHeader>
						<TableBody>
							{data.recent_runs.map((run) => (
								<TableRow key={run.id}>
									<TableCell>
										<div className="flex flex-col gap-1">
											<span>{run.mode}</span>
											<span className="font-mono text-xs text-muted-foreground">
												{run.index_version ?? "no index"}
											</span>
										</div>
									</TableCell>
									<TableCell>
										<RunStatus run={run} />
									</TableCell>
									<TableCell className="tabular-nums">
										{run.metrics ? run.metrics.ndcg_at_10.toFixed(3) : "—"}
									</TableCell>
									<TableCell className="tabular-nums">
										{run.metrics ? `${(run.metrics.judged_at_10 * 100).toFixed(0)}%` : "—"}
									</TableCell>
									<TableCell className="tabular-nums">
										{run.metrics ? `${run.metrics.latency_p95_ms.toFixed(0)} ms` : "—"}
									</TableCell>
									<TableCell>
										<div className="flex justify-end gap-1">
											{run.status === "needs_judgments" && run.missing_judgments > 0 ? (
												<Button
													size="sm"
													variant="ghost"
													nativeButton={false}
													render={<Link to="/admin/evals/judge" />}
												>
													<ArrowRight data-icon="inline-end" />
													Review {run.missing_judgments}
												</Button>
											) : null}
											{run.status === "needs_judgments" && run.missing_judgments === 0 ? (
												<Button
													size="sm"
													variant="ghost"
													disabled={finalize.isPending}
													onClick={() => finalize.mutate(run.id)}
												>
													<RefreshCw data-icon="inline-start" />
													Recalculate
												</Button>
											) : null}
											{run.status === "complete" && data.baseline_run_id !== run.id ? (
												<Button
													size="sm"
													variant="ghost"
													disabled={baseline.isPending}
													onClick={() => baseline.mutate(run.id)}
												>
													<Baseline data-icon="inline-start" />
													Set baseline
												</Button>
											) : null}
											{run.status === "complete" &&
											data.baseline_run_id &&
											data.baseline_run_id !== run.id ? (
												<Button
													size="sm"
													variant="ghost"
													nativeButton={false}
													render={
														<Link
															to="/admin/evals/compare"
															search={{ baseline: data.baseline_run_id, candidate: run.id }}
														/>
													}
												>
													Compare <ArrowRight data-icon="inline-end" />
												</Button>
											) : null}
										</div>
									</TableCell>
								</TableRow>
							))}
						</TableBody>
					</Table>
				</div>
			)}
		</section>
	);
}

function RunStatus({ run }: { run: SearchEvalRun }) {
	const isActive = run.status === "queued" || run.status === "running";
	const progress = run.progress_total ? (run.progress_completed / run.progress_total) * 100 : 0;
	return (
		<div className="flex min-w-32 flex-col gap-2">
			<Badge
				variant={
					run.status === "failed"
						? "destructive"
						: run.status === "complete"
							? "outline"
							: "secondary"
				}
			>
				{run.phase ?? run.status.replace("_", " ")}
			</Badge>
			{isActive ? (
				<div className="flex items-center gap-2">
					<Progress
						value={progress}
						aria-label={`${run.progress_completed} of ${run.progress_total} queries`}
					/>
					<span className="text-xs text-muted-foreground tabular-nums">
						{run.progress_completed}/{run.progress_total}
					</span>
				</div>
			) : null}
			{run.error ? <span className="max-w-56 text-xs text-destructive">{run.error}</span> : null}
		</div>
	);
}

function useOverviewData() {
	return useQuery(searchEvalOverviewQueryOptions()).data;
}

function OverviewSkeleton() {
	return (
		<div className="flex flex-col gap-6">
			<div className="grid gap-4 lg:grid-cols-2">
				<Skeleton className="h-44" />
				<Skeleton className="h-44" />
			</div>
			<Skeleton className="h-72" />
		</div>
	);
}
