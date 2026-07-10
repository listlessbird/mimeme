import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
	type IngestionRow,
	INGESTION_POLL_MS,
	ingestionQueryOptions,
	sourcesQueryOptions,
} from "@/lib/admin/api";
import { relativeTime } from "@/lib/admin/format";
import {
	INGEST_OUTCOMES,
	INGEST_STAGES,
	INGEST_TRIGGERS,
	INGESTION_PAGE_SIZE,
	type IngestionView,
	ingestionSearchParsers,
	isLiveView,
	pageToOffset,
} from "@/lib/admin/ingestion-search";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useQueryStates } from "nuqs";
import { useMemo } from "react";

import { IngestOutcomeBadge, StageBadge, TriggerBadge } from "./badges";
import { FilterSelect } from "./filter-select";
import { ItemThumbnail } from "./item-thumbnail";

const VIEWS: { value: IngestionView; label: string }[] = [
	{ value: "live", label: "live" },
	{ value: "completed", label: "completed" },
	{ value: "failed", label: "failed" },
	{ value: "all", label: "all" },
];

const IN_FLIGHT_STATUSES = new Set<IngestionRow["status"]>(["PENDING", "RUNNING"]);

export function IngestionBrowse() {
	const [filters, setFilters] = useQueryStates(ingestionSearchParsers);
	const live = isLiveView(filters.view);

	const offset = pageToOffset(filters.page);
	const params = {
		view: filters.view,
		stage: filters.stage,
		trigger: filters.trigger,
		source_id: filters.source_id,
		dataset: filters.dataset,
		outcome: filters.outcome,
		created_from: filters.from ? startOfDayIso(filters.from) : null,
		created_to: filters.to ? endOfDayIso(filters.to) : null,
		limit: INGESTION_PAGE_SIZE,
		offset,
	};

	const { data, isPending, isFetching } = useQuery({
		...ingestionQueryOptions(params, { poll: live }),
		placeholderData: keepPreviousData,
		throwOnError: true,
	});

	const { data: sources } = useQuery(sourcesQueryOptions());
	const groups = useMemo(() => groupRows(data?.rows ?? []), [data?.rows]);

	const resetPage = { page: 1 };

	return (
		<div className="flex flex-col gap-4">
			<div className="flex flex-wrap items-center justify-between gap-3">
				<Tabs
					value={filters.view}
					onValueChange={(value) => setFilters({ view: value as IngestionView, ...resetPage })}
				>
					<TabsList>
						{VIEWS.map((v) => (
							<TabsTrigger key={v.value} value={v.value}>
								{v.label}
							</TabsTrigger>
						))}
					</TabsList>
				</Tabs>
				<LiveIndicator active={live && isFetching} />
			</div>

			<div className="flex flex-wrap items-end gap-3">
				<Field label="stage">
					<FilterSelect
						className="h-9 w-40"
						value={filters.stage ?? ""}
						onValueChange={(v) => setFilters({ stage: enumOrNull(v, INGEST_STAGES), ...resetPage })}
						options={INGEST_STAGES.map((s) => ({ value: s, label: s.toLowerCase() }))}
					/>
				</Field>
				<Field label="trigger">
					<FilterSelect
						className="h-9 w-40"
						value={filters.trigger ?? ""}
						onValueChange={(v) =>
							setFilters({ trigger: enumOrNull(v, INGEST_TRIGGERS), ...resetPage })
						}
						options={INGEST_TRIGGERS.map((t) => ({ value: t, label: t }))}
					/>
				</Field>
				<Field label="source">
					<FilterSelect
						className="h-9 w-44"
						value={filters.source_id ? String(filters.source_id) : ""}
						onValueChange={(v) => setFilters({ source_id: v ? Number(v) : null, ...resetPage })}
						options={(sources?.sources ?? []).map((s) => ({
							value: String(s.id),
							label: s.name,
						}))}
					/>
				</Field>
				<Field label="outcome">
					<FilterSelect
						className="h-9 w-40"
						value={filters.outcome ?? ""}
						onValueChange={(v) =>
							setFilters({ outcome: enumOrNull(v, INGEST_OUTCOMES), ...resetPage })
						}
						options={INGEST_OUTCOMES.map((o) => ({ value: o, label: o }))}
					/>
				</Field>
				<Field label="dataset">
					<Input
						value={filters.dataset ?? ""}
						placeholder="any"
						maxLength={255}
						className="h-9 w-40"
						onChange={(e) => setFilters({ dataset: e.target.value || null, ...resetPage })}
					/>
				</Field>
				<Field label="from">
					<Input
						type="date"
						value={toDateInput(filters.from)}
						className="h-9 w-40"
						onChange={(e) =>
							setFilters({ from: e.target.value ? new Date(e.target.value) : null, ...resetPage })
						}
					/>
				</Field>
				<Field label="to">
					<Input
						type="date"
						value={toDateInput(filters.to)}
						className="h-9 w-40"
						onChange={(e) =>
							setFilters({ to: e.target.value ? new Date(e.target.value) : null, ...resetPage })
						}
					/>
				</Field>
			</div>

			{isPending ? (
				<IngestionSkeleton />
			) : !data || data.rows.length === 0 ? (
				<Empty className="border">
					<EmptyHeader>
						<EmptyTitle>nothing here</EmptyTitle>
						<EmptyDescription>
							{live
								? "no memes are moving through the pipeline right now."
								: "no ingest attempts match these filters."}
						</EmptyDescription>
					</EmptyHeader>
				</Empty>
			) : (
				<div className="flex flex-col gap-5">
					{groups.map((group) => (
						<GroupCard key={group.key} group={group} />
					))}
				</div>
			)}

			<PageBar
				page={filters.page}
				pageSize={INGESTION_PAGE_SIZE}
				total={data?.total ?? 0}
				loaded={data?.rows.length ?? 0}
				offset={offset}
				isFetching={isFetching}
				onPageChange={(page) => setFilters({ page })}
			/>
		</div>
	);
}

interface RowGroup {
	key: string;
	jobId: string;
	sourceId: number | null;
	sourceName: string | null;
	sourceRunId: number | null;
	trigger: IngestionRow["trigger"];
	rows: IngestionRow[];
}

function groupRows(rows: IngestionRow[]): RowGroup[] {
	const groups: RowGroup[] = [];
	const byKey = new Map<string, RowGroup>();

	for (const row of rows) {
		const key = row.source_run_id != null ? `run-${row.source_run_id}` : `job-${row.job_id}`;
		let group = byKey.get(key);
		if (!group) {
			group = {
				key,
				jobId: row.job_id,
				sourceId: row.source_id,
				sourceName: row.source_name,
				sourceRunId: row.source_run_id,
				trigger: row.trigger,
				rows: [],
			};
			byKey.set(key, group);
			groups.push(group);
		}
		group.rows.push(row);
	}

	return groups;
}

function GroupCard({ group }: { group: RowGroup }) {
	const done = group.rows.filter((r) => r.outcome === "ingested" || r.outcome === "deduped").length;

	return (
		<div className="flex flex-col gap-2 rounded-lg border">
			<div className="flex flex-wrap items-center gap-2 border-b bg-muted/30 px-3 py-2">
				<TriggerBadge trigger={group.trigger} />
				{group.sourceName ? (
					group.sourceId != null ? (
						<Link
							to="/admin/sources/$id"
							params={{ id: String(group.sourceId) }}
							className="text-sm font-medium hover:underline"
						>
							{group.sourceName}
						</Link>
					) : (
						<span className="text-sm font-medium">{group.sourceName}</span>
					)
				) : (
					<span className="text-sm font-medium text-muted-foreground">manual upload</span>
				)}
				<span className="font-mono text-xs text-muted-foreground">
					{group.sourceRunId != null ? `run #${group.sourceRunId}` : group.jobId}
				</span>
				<Badge variant="outline" className="ml-auto tabular-nums">
					{done}/{group.rows.length} done
				</Badge>
			</div>
			<ul className="divide-y">
				{group.rows.map((row) => (
					<IngestionRowItem key={row.ingest_url_id} row={row} />
				))}
			</ul>
		</div>
	);
}

function IngestionRowItem({ row }: { row: IngestionRow }) {
	const inFlight = IN_FLIGHT_STATUSES.has(row.status);
	const failed = row.status === "FAILED";

	return (
		<li>
			<Link
				to="/admin/ingestion/$ingestUrlId"
				params={{ ingestUrlId: String(row.ingest_url_id) }}
				className="flex items-center gap-3 px-3 py-2 transition-colors hover:bg-accent/40 focus-visible:bg-accent/40 focus-visible:outline-none"
			>
				<ItemThumbnail
					src={row.thumbnail_url ?? null}
					alt={`attempt ${row.ingest_url_id}`}
					className="size-12 shrink-0"
				/>
				<div className="flex min-w-0 flex-1 flex-col gap-1">
					<div className="flex flex-wrap items-center gap-2">
						{inFlight || failed ? <StageBadge stage={row.stage} frozen={failed} /> : null}
						<IngestOutcomeBadge outcome={row.outcome} />
						<span className="font-mono text-xs text-muted-foreground">#{row.ingest_url_id}</span>
						{row.resolved_image_id != null ? (
							<span className="font-mono text-xs text-muted-foreground">
								→ image #{row.resolved_image_id}
							</span>
						) : null}
					</div>
					{failed && row.error_message ? (
						<p className="truncate text-xs text-destructive">{row.error_message}</p>
					) : (
						<p className="truncate text-xs text-muted-foreground">{row.url}</p>
					)}
				</div>
				<div className="hidden shrink-0 text-right text-xs text-muted-foreground sm:block">
					{inFlight && row.stage_updated_at ? (
						<span title={row.stage_updated_at}>in stage {relativeTime(row.stage_updated_at)}</span>
					) : (
						<span title={row.created_at}>{relativeTime(row.created_at)}</span>
					)}
				</div>
			</Link>
		</li>
	);
}

function LiveIndicator({ active }: { active: boolean }) {
	return (
		<span
			className="relative flex size-2"
			title={active ? `polling every ${INGESTION_POLL_MS / 1000}s` : "static"}
		>
			{active ? (
				<span className="absolute inline-flex size-full animate-ping rounded-full bg-emerald-500/70" />
			) : null}
			<span
				className={`relative inline-flex size-2 rounded-full ${active ? "bg-emerald-500" : "bg-muted-foreground/40"}`}
			/>
		</span>
	);
}

function PageBar({
	page,
	pageSize,
	total,
	loaded,
	offset,
	isFetching,
	onPageChange,
}: {
	page: number;
	pageSize: number;
	total: number;
	loaded: number;
	offset: number;
	isFetching?: boolean;
	onPageChange: (page: number) => void;
}) {
	const from = total === 0 ? 0 : offset + 1;
	const to = offset + loaded;
	const hasPrev = page > 1;
	const hasNext = offset + pageSize < total;

	return (
		<div className="flex items-center justify-between">
			<p className="text-xs text-muted-foreground tabular-nums">
				{from}–{to} of {total}
			</p>
			<div className="flex items-center gap-2">
				<Button
					variant="outline"
					size="sm"
					disabled={!hasPrev || isFetching}
					onClick={() => onPageChange(Math.max(1, page - 1))}
				>
					<ChevronLeft data-icon="inline-start" />
					prev
				</Button>
				<Button
					variant="outline"
					size="sm"
					disabled={!hasNext || isFetching}
					onClick={() => onPageChange(page + 1)}
				>
					next
					<ChevronRight data-icon="inline-end" />
				</Button>
			</div>
		</div>
	);
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
	return (
		<div className="flex flex-col gap-1.5">
			<Label className="text-xs text-muted-foreground">{label}</Label>
			{children}
		</div>
	);
}

function IngestionSkeleton() {
	return (
		<div className="flex flex-col gap-3">
			{Array.from({ length: 3 }).map((_, g) => (
				<div key={g} className="flex flex-col gap-2 rounded-lg border p-3">
					<Skeleton className="h-5 w-48" />
					{Array.from({ length: 3 }).map((__, i) => (
						<Skeleton key={i} className="h-12 w-full" />
					))}
				</div>
			))}
		</div>
	);
}

function enumOrNull<T extends string>(value: string, allowed: readonly T[]): T | null {
	return allowed.includes(value as T) ? (value as T) : null;
}

function toDateInput(date: Date | null): string {
	if (!date) return "";
	return date.toISOString().slice(0, 10);
}

function startOfDayIso(date: Date): string {
	const d = new Date(date);
	d.setHours(0, 0, 0, 0);
	return d.toISOString();
}

function endOfDayIso(date: Date): string {
	const d = new Date(date);
	d.setHours(23, 59, 59, 999);
	return d.toISOString();
}
