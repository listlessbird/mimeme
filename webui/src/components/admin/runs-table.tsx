import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from "@/components/ui/table";
import type { SourceRun } from "@/lib/admin/api";
import { relativeTime } from "@/lib/admin/format";
import { Link } from "@tanstack/react-router";

import { RunStatusBadge, TriggerBadge } from "./badges";
import { RawJsonDrawer } from "./dev-toolkit";
import { JobIdChip } from "./job-detail";

export function RunsTable({ sourceId, runs }: { sourceId: number; runs: SourceRun[] }) {
	if (runs.length === 0) {
		return (
			<Empty className="border">
				<EmptyHeader>
					<EmptyTitle>no runs yet</EmptyTitle>
					<EmptyDescription>
						trigger a run, or wait for the schedule — runs appear here automatically.
					</EmptyDescription>
				</EmptyHeader>
			</Empty>
		);
	}

	return (
		<div className="rounded-md border">
			<Table>
				<TableHeader>
					<TableRow>
						<TableHead>status</TableHead>
						<TableHead>trigger</TableHead>
						<TableHead>started</TableHead>
						<TableHead className="text-right">discovered</TableHead>
						<TableHead className="text-right">queued</TableHead>
						<TableHead className="text-right">duplicate</TableHead>
						<TableHead className="text-right">failed</TableHead>
						<TableHead className="text-right">items</TableHead>
					</TableRow>
				</TableHeader>
				<TableBody>
					{runs.map((run) => (
						<TableRow key={run.id}>
							<TableCell>
								<RunStatusBadge status={run.status} />
							</TableCell>
							<TableCell>
								<TriggerBadge trigger={run.trigger_mode} />
							</TableCell>
							<TableCell className="text-muted-foreground">
								{relativeTime(run.started_at ?? run.created_at)}
							</TableCell>
							<TableCell className="text-right tabular-nums">{run.discovered}</TableCell>
							<TableCell className="text-right tabular-nums">{run.queued}</TableCell>
							<TableCell className="text-right tabular-nums">{run.duplicate}</TableCell>
							<TableCell className="text-right tabular-nums">{run.failed}</TableCell>
							<TableCell className="text-right">
								<div className="flex items-center justify-end gap-2">
									{run.ingest_job_id ? <JobIdChip jobId={run.ingest_job_id} /> : null}
									<RawJsonDrawer data={run} title={`run #${run.id}`} />
									<Link
										to="/admin/sources/$id/runs/$runId"
										params={{ id: String(sourceId), runId: String(run.id) }}
										className="text-sm hover:underline"
									>
										view
									</Link>
								</div>
							</TableCell>
						</TableRow>
					))}
				</TableBody>
			</Table>
		</div>
	);
}
