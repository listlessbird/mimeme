import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { type IngestionLogEntry, ingestionLogsQueryOptions } from "@/lib/admin/api";
import { absoluteTime } from "@/lib/admin/format";
import { cn } from "@/lib/utils";
import { useQuery } from "@tanstack/react-query";

import { IdChip } from "./dev-toolkit";

export function IngestionLogs({ ingestUrlId, poll }: { ingestUrlId: number; poll: boolean }) {
	const { data, isPending } = useQuery(ingestionLogsQueryOptions(ingestUrlId, { poll }));

	return (
		<Card>
			<CardHeader>
				<CardTitle className="flex flex-wrap items-center gap-2 text-sm">
					logs
					{data?.workflow_id ? (
						<IdChip label="workflow_id" value={data.workflow_id} truncate />
					) : null}
				</CardTitle>
			</CardHeader>
			<CardContent>
				{isPending ? (
					<div className="flex flex-col gap-2">
						{Array.from({ length: 4 }).map((_, i) => (
							<Skeleton key={i} className="h-5 w-full" />
						))}
					</div>
				) : !data?.available ? (
					<p className="text-sm text-muted-foreground">
						log streaming isn't configured — set <code className="font-mono">AXIOM_API_TOKEN</code>{" "}
						and <code className="font-mono">AXIOM_DATASET</code> to surface pipeline events here.
					</p>
				) : data.entries.length === 0 ? (
					<p className="text-sm text-muted-foreground">no events recorded for this attempt yet.</p>
				) : (
					<ScrollArea className="max-h-80">
						<ol className="flex flex-col gap-1.5 pr-3">
							{data.entries.map((entry, i) => (
								<LogRow key={`${entry.time}-${i}`} entry={entry} />
							))}
						</ol>
					</ScrollArea>
				)}
			</CardContent>
		</Card>
	);
}

function LogRow({ entry }: { entry: IngestionLogEntry }) {
	const isError = entry.outcome === "error" || entry.level === "error" || entry.error != null;
	const label = entry.activity_name ?? entry.step ?? entry.event ?? "event";

	return (
		<li
			className={cn(
				"flex flex-col gap-1 rounded-md border px-2.5 py-1.5 text-xs",
				isError && "border-destructive/40 bg-destructive/5",
			)}
		>
			<div className="flex flex-wrap items-center gap-2">
				<span className="font-mono text-muted-foreground tabular-nums" title={entry.time}>
					{absoluteTime(entry.time)}
				</span>
				<span className="font-mono">{label}</span>
				{entry.outcome ? (
					<Badge variant={isError ? "destructive" : "secondary"} className="py-0">
						{entry.outcome}
					</Badge>
				) : null}
				{entry.duration_ms != null ? (
					<span className="text-muted-foreground tabular-nums">{entry.duration_ms}ms</span>
				) : null}
				{entry.attempt != null && entry.attempt > 1 ? (
					<span className="text-muted-foreground">retry #{entry.attempt}</span>
				) : null}
			</div>
			{entry.error ? (
				<p className="font-mono break-words whitespace-pre-wrap text-destructive">{entry.error}</p>
			) : null}
		</li>
	);
}
