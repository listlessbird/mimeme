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
import {
	Select,
	SelectContent,
	SelectGroup,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from "@/components/ui/table";
import {
	searchEvalComparisonQueryOptions,
	searchEvalOverviewQueryOptions,
	type SearchEvalComparison as ComparisonData,
} from "@/lib/admin/search-eval-api";
import { cn } from "@/lib/utils";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { ArrowDown, ArrowRight, Equal, ImageOff } from "lucide-react";
import { useState } from "react";

/** Side-by-side comparison of two runs scored from the same judgment snapshot. */
export function SearchEvalComparison({
	baselineRunId,
	candidateRunId,
}: {
	baselineRunId?: string;
	candidateRunId?: string;
}) {
	const overview = useQuery(searchEvalOverviewQueryOptions());
	const baseline = baselineRunId ?? overview.data?.baseline_run_id ?? "";
	const candidate = candidateRunId ?? "";
	const comparison = useQuery(searchEvalComparisonQueryOptions(baseline, candidate));
	if (overview.isPending) return <ComparisonSkeleton />;
	const completeRuns = overview.data?.recent_runs.filter((run) => run.status === "complete") ?? [];
	return (
		<div className="flex flex-col gap-6">
			<ComparisonPicker baseline={baseline} candidate={candidate} runs={completeRuns} />
			{!baseline || !candidate ? (
				<Empty className="min-h-72 border">
					<EmptyHeader>
						<EmptyTitle>Choose two complete runs</EmptyTitle>
						<EmptyDescription>
							The baseline and candidate must use the same query set.
						</EmptyDescription>
					</EmptyHeader>
				</Empty>
			) : comparison.isPending ? (
				<ComparisonSkeleton />
			) : comparison.data ? (
				<ComparisonResults key={`${baseline}:${candidate}`} comparison={comparison.data} />
			) : null}
		</div>
	);
}

function ComparisonPicker({
	baseline,
	candidate,
	runs,
}: {
	baseline: string;
	candidate: string;
	runs: Array<{ id: string; mode: string; index_version: string | null }>;
}) {
	const navigate = useNavigate();
	const update = (next: { baseline?: string; candidate?: string }) =>
		navigate({
			to: "/admin/evals/compare",
			search: { baseline: next.baseline ?? baseline, candidate: next.candidate ?? candidate },
			replace: true,
		});
	return (
		<div className="grid gap-4 border bg-card p-4 md:grid-cols-[1fr_auto_1fr] md:items-end md:p-5">
			<Field>
				<FieldLabel htmlFor="baseline-run">Baseline</FieldLabel>
				<Select
					value={baseline || null}
					onValueChange={(value) => update({ baseline: value ?? "" })}
				>
					<SelectTrigger id="baseline-run" className="w-full">
						<SelectValue placeholder="Select a run" />
					</SelectTrigger>
					<SelectContent>
						<SelectGroup>
							{runs.map((run) => (
								<SelectItem key={run.id} value={run.id}>
									{runLabel(run)}
								</SelectItem>
							))}
						</SelectGroup>
					</SelectContent>
				</Select>
			</Field>
			<ArrowRight className="mb-2 hidden text-muted-foreground md:block" />
			<Field>
				<FieldLabel htmlFor="candidate-run">Candidate</FieldLabel>
				<Select
					value={candidate || null}
					onValueChange={(value) => update({ candidate: value ?? "" })}
				>
					<SelectTrigger id="candidate-run" className="w-full">
						<SelectValue placeholder="Select a run" />
					</SelectTrigger>
					<SelectContent>
						<SelectGroup>
							{runs.map((run) =>
								run.id === baseline ? null : (
									<SelectItem key={run.id} value={run.id}>
										{runLabel(run)}
									</SelectItem>
								),
							)}
						</SelectGroup>
					</SelectContent>
				</Select>
			</Field>
		</div>
	);
}

function runLabel(run: { id: string; mode: string; index_version: string | null }): string {
	return `${run.mode} · ${run.index_version ?? "unknown index"} · ${run.id.slice(0, 7)}`;
}

function ComparisonResults({ comparison }: { comparison: ComparisonData }) {
	const [selectedQueryId, setSelectedQueryId] = useState(comparison.queries[0]?.query_id ?? null);
	const selected =
		comparison.queries.find((query) => query.query_id === selectedQueryId) ?? comparison.queries[0];
	return (
		<div className="flex flex-col gap-6">
			<MetricComparison comparison={comparison} />
			<div className="grid gap-4 xl:grid-cols-[21rem_minmax(0,1fr)]">
				<Card className="h-fit">
					<CardHeader>
						<CardTitle>Query deltas</CardTitle>
						<CardDescription>Worst regressions first.</CardDescription>
						<CardAction>
							<Badge variant="secondary">{comparison.queries.length}</Badge>
						</CardAction>
					</CardHeader>
					<CardContent className="flex max-h-[50rem] flex-col gap-1 overflow-y-auto">
						{comparison.queries.map((query) => (
							<Button
								key={query.query_id}
								variant={query.query_id === selected?.query_id ? "secondary" : "ghost"}
								className="h-auto justify-between py-2 text-left whitespace-normal"
								onClick={() => setSelectedQueryId(query.query_id)}
							>
								<span className="line-clamp-2 flex-1">{query.text}</span>
								<Delta value={query.delta_ndcg_at_10} format="compact-score" />
							</Button>
						))}
					</CardContent>
				</Card>
				{selected ? <RankingDifference query={selected} /> : null}
			</div>
		</div>
	);
}

function MetricComparison({ comparison }: { comparison: ComparisonData }) {
	const baseline = comparison.baseline.metrics;
	const candidate = comparison.candidate.metrics;
	if (!baseline || !candidate) return null;
	const rows = [
		["nDCG@10", baseline.ndcg_at_10, candidate.ndcg_at_10, comparison.delta_ndcg_at_10, "score"],
		[
			"Success@5",
			baseline.success_at_5,
			candidate.success_at_5,
			comparison.delta_success_at_5,
			"percent",
		],
		[
			"Precision@5",
			baseline.precision_at_5,
			candidate.precision_at_5,
			candidate.precision_at_5 - baseline.precision_at_5,
			"percent",
		],
		["MRR@10", baseline.mrr_at_10, candidate.mrr_at_10, comparison.delta_mrr_at_10, "score"],
		[
			"Judged@10",
			baseline.judged_at_10,
			candidate.judged_at_10,
			candidate.judged_at_10 - baseline.judged_at_10,
			"percent",
		],
		[
			"p95 latency",
			baseline.latency_p95_ms,
			candidate.latency_p95_ms,
			comparison.delta_latency_p95_ms,
			"latency",
		],
	] as const;
	return (
		<div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_18rem]">
			<div className="border">
				<Table>
					<TableHeader>
						<TableRow>
							<TableHead>Metric</TableHead>
							<TableHead>Baseline</TableHead>
							<TableHead>Candidate</TableHead>
							<TableHead>Delta</TableHead>
						</TableRow>
					</TableHeader>
					<TableBody>
						{rows.map(([label, before, after, delta, format]) => (
							<TableRow key={label}>
								<TableCell className="font-medium">{label}</TableCell>
								<TableCell className="tabular-nums">{formatMetric(before, format)}</TableCell>
								<TableCell className="tabular-nums">{formatMetric(after, format)}</TableCell>
								<TableCell>
									<Delta value={delta} format={format === "latency" ? "latency" : "score"} />
								</TableCell>
							</TableRow>
						))}
					</TableBody>
				</Table>
			</div>
			<Card>
				<CardHeader>
					<CardTitle>Query movement</CardTitle>
					<CardDescription>Paired by the same search need.</CardDescription>
				</CardHeader>
				<CardContent className="flex flex-col gap-3 text-sm">
					<Movement label="improved" value={comparison.improved_queries} />
					<Movement label="unchanged" value={comparison.unchanged_queries} />
					<Movement label="regressed" value={comparison.regressed_queries} />
				</CardContent>
			</Card>
		</div>
	);
}

function Movement({ label, value }: { label: string; value: number }) {
	return (
		<div className="flex items-center justify-between">
			<span className="text-muted-foreground">{label}</span>
			<span className="font-semibold tabular-nums">{value}</span>
		</div>
	);
}

function formatMetric(value: number, format: "score" | "percent" | "latency"): string {
	if (format === "percent") return `${(value * 100).toFixed(1)}%`;
	if (format === "latency") return `${value.toFixed(0)} ms`;
	return value.toFixed(3);
}

type DeltaFormat = "score" | "compact-score" | "latency";

function Delta({ value, format = "score" }: { value: number; format?: DeltaFormat }) {
	const latency = format === "latency";
	const isNeutral = Math.abs(value) < 0.0005;
	const isGood = latency ? value < 0 : value > 0;
	const label = latency
		? `${value > 0 ? "+" : ""}${value.toFixed(0)} ms`
		: `${value > 0 ? "+" : ""}${value.toFixed(format === "compact-score" ? 2 : 3)}`;
	return (
		<Badge variant={isNeutral ? "secondary" : isGood ? "default" : "destructive"}>
			{isNeutral ? <Equal /> : value < 0 ? <ArrowDown /> : null}
			{label}
		</Badge>
	);
}

function RankingDifference({ query }: { query: ComparisonData["queries"][number] }) {
	return (
		<section className="flex min-w-0 flex-col gap-4">
			<div className="flex flex-col gap-2 border bg-card p-4">
				<div className="flex items-center gap-2">
					<Badge variant="outline">{query.intent}</Badge>
					<Delta value={query.delta_ndcg_at_10} />
				</div>
				<h2 className="text-lg leading-snug font-semibold">{query.text}</h2>
			</div>
			<div className="grid gap-4 md:grid-cols-2">
				<RankingColumn
					title="Baseline"
					score={query.baseline_ndcg_at_10}
					results={query.baseline_results}
				/>
				<RankingColumn
					title="Candidate"
					score={query.candidate_ndcg_at_10}
					results={query.candidate_results}
				/>
			</div>
		</section>
	);
}

function RankingColumn({
	title,
	score,
	results,
}: {
	title: string;
	score: number;
	results: ComparisonData["queries"][number]["baseline_results"];
}) {
	return (
		<Card>
			<CardHeader>
				<CardTitle>{title}</CardTitle>
				<CardDescription>nDCG@10 {score.toFixed(3)}</CardDescription>
			</CardHeader>
			<CardContent className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-2 lg:grid-cols-3">
				{results.map((result) => (
					<div
						key={result.image_id}
						className="relative aspect-square overflow-hidden border bg-muted"
					>
						{result.url ? (
							<img
								src={result.url}
								alt={`Rank ${result.rank}, grade ${result.grade ?? "unjudged"}`}
								className="size-full object-cover outline outline-1 -outline-offset-1 outline-black/10 dark:outline-white/10"
								loading="lazy"
							/>
						) : (
							<div className="flex size-full items-center justify-center text-muted-foreground">
								<ImageOff />
							</div>
						)}
						<Badge variant="secondary" className="absolute top-1 left-1">
							#{result.rank}
						</Badge>
						<Badge
							variant={
								result.grade !== null && result.grade !== undefined && result.grade >= 2
									? "default"
									: "secondary"
							}
							className={cn("absolute right-1 bottom-1", result.grade == null && "opacity-70")}
						>
							{result.grade ?? "U"}
						</Badge>
					</div>
				))}
			</CardContent>
		</Card>
	);
}

function ComparisonSkeleton() {
	return (
		<div className="flex flex-col gap-4">
			<Skeleton className="h-24" />
			<Skeleton className="h-56" />
			<Skeleton className="h-[36rem]" />
		</div>
	);
}
