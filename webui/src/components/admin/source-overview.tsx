import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { SourceDetail } from "@/lib/admin/api";
import { absoluteTime, scheduleLabel } from "@/lib/admin/format";

function Stat({ label, value }: { label: string; value: number }) {
	return (
		<Card>
			<CardHeader>
				<CardTitle className="text-xs font-medium text-muted-foreground">{label}</CardTitle>
			</CardHeader>
			<CardContent>
				<p className="text-2xl font-semibold tabular-nums">{value}</p>
			</CardContent>
		</Card>
	);
}

function ConfigRow({ label, children }: { label: string; children: React.ReactNode }) {
	return (
		<div className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:gap-4">
			<dt className="w-40 shrink-0 text-sm text-muted-foreground">{label}</dt>
			<dd className="text-sm">{children}</dd>
		</div>
	);
}

export function SourceOverview({ source }: { source: SourceDetail }) {
	const subreddits = source.adapter_config?.subreddits ?? [];

	return (
		<div className="flex flex-col gap-6">
			<div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
				<Stat label="runs" value={source.stats.run_count} />
				<Stat label="items discovered" value={source.stats.items_discovered} />
				<Stat label="duplicates" value={source.stats.duplicate_count} />
			</div>

			<Card>
				<CardHeader>
					<CardTitle>configuration</CardTitle>
				</CardHeader>
				<CardContent>
					<dl className="flex flex-col gap-3">
						<ConfigRow label="adapter">{source.adapter_key}</ConfigRow>
						<ConfigRow label="subreddits">
							{subreddits.length ? (
								<div className="flex flex-wrap gap-1.5">
									{subreddits.map((subreddit) => (
										<Badge key={subreddit} variant="secondary">
											r/{subreddit}
										</Badge>
									))}
								</div>
							) : (
								<span className="text-muted-foreground">none</span>
							)}
						</ConfigRow>
						<ConfigRow label="schedule">
							{scheduleLabel(source.schedule_cron, source.schedule_timezone)}
						</ConfigRow>
						<ConfigRow label="max items / run">
							{source.max_items_per_run ?? "adapter default"}
						</ConfigRow>
						<ConfigRow label="dataset">
							{source.dataset ?? <span className="text-muted-foreground">none</span>}
						</ConfigRow>
						<ConfigRow label="created">{absoluteTime(source.created_at)}</ConfigRow>
						<ConfigRow label="updated">{absoluteTime(source.updated_at)}</ConfigRow>
					</dl>
				</CardContent>
			</Card>
		</div>
	);
}
