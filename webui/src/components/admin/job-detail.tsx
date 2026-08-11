import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
	Sheet,
	SheetContent,
	SheetDescription,
	SheetHeader,
	SheetTitle,
	SheetTrigger,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { type Job, JOB_POLL_MS, jobQueryOptions } from "@/lib/admin/api";
import { absoluteTime, relativeTime } from "@/lib/admin/format";
import { useQuery } from "@tanstack/react-query";
import { ArrowUpRight } from "lucide-react";
import { useState } from "react";

import { JobStatusBadge } from "./badges";
import { CopyButton, RawJsonDrawer } from "./dev-toolkit";

const ACTIVE: ReadonlySet<Job["status"]> = new Set<Job["status"]>(["PENDING", "RUNNING"]);

export function JobIdChip({ jobId, label = "job_id" }: { jobId: string; label?: string }) {
	const [open, setOpen] = useState(false);

	return (
		<Sheet open={open} onOpenChange={setOpen}>
			<Badge
				variant="outline"
				className="gap-1 py-0 pr-0.5 pl-2 font-normal"
				title={`${label}: ${jobId}`}
			>
				<span className="text-muted-foreground">{label}</span>
				<SheetTrigger
					render={
						<button
							type="button"
							aria-label={`view details for ${label}`}
							className="inline-flex items-center gap-0.5 font-mono hover:underline"
						/>
					}
				>
					<span className="max-w-[16ch] truncate">{jobId}</span>
					<ArrowUpRight className="size-3 shrink-0" />
				</SheetTrigger>
				<CopyButton value={jobId} label={label} className="size-5 text-muted-foreground" />
			</Badge>
			<SheetContent className="w-full gap-0 overflow-y-auto sm:max-w-lg">
				<JobDetailBody jobId={jobId} enabled={open} />
			</SheetContent>
		</Sheet>
	);
}

function JobDetailBody({ jobId, enabled }: { jobId: string; enabled: boolean }) {
	const {
		data: job,
		isPending,
		isError,
	} = useQuery({
		...jobQueryOptions(jobId),
		enabled,
		refetchInterval: (query) => {
			const status = query.state.data?.status;
			return status !== undefined && ACTIVE.has(status) ? JOB_POLL_MS : false;
		},
	});

	return (
		<>
			<SheetHeader>
				<SheetTitle className="font-mono text-sm break-all">{jobId}</SheetTitle>
				<SheetDescription>ingest / index job — status, timing, and result</SheetDescription>
			</SheetHeader>

			{isPending ? (
				<div className="flex flex-col gap-3 px-4">
					<Skeleton className="h-6 w-40" />
					<Skeleton className="h-2 w-full" />
					<Skeleton className="h-24 w-full" />
				</div>
			) : isError || !job ? (
				<p className="px-4 text-sm text-muted-foreground">could not load this job.</p>
			) : (
				<div className="flex flex-col gap-4 px-4 pb-6">
					<div className="flex flex-wrap items-center gap-2">
						<JobStatusBadge status={job.status} />
						<Badge variant="outline">{job.type}</Badge>
						<RawJsonDrawer data={job} title={`job ${job.id}`} />
					</div>

					<div className="flex flex-col gap-1.5">
						<div className="flex items-center justify-between text-xs text-muted-foreground tabular-nums">
							<span>progress</span>
							<span>{Math.round(job.progress)}%</span>
						</div>
						<Progress value={job.progress} />
					</div>

					{job.message ? (
						<p className="rounded-md bg-muted p-3 text-sm break-words">{job.message}</p>
					) : null}

					<dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
						<Row label="created">
							<span title={absoluteTime(job.created_at)}>{relativeTime(job.created_at)}</span>
						</Row>
						<Row label="started">
							<span title={absoluteTime(job.started_at ?? null)}>
								{relativeTime(job.started_at ?? null)}
							</span>
						</Row>
						<Row label="completed">
							<span title={absoluteTime(job.completed_at ?? null)}>
								{relativeTime(job.completed_at ?? null)}
							</span>
						</Row>
					</dl>

					<div className="flex flex-col gap-2">
						<p className="text-xs font-medium text-muted-foreground">result</p>
						{job.result ? (
							<pre className="rounded-md bg-muted p-3 font-mono text-xs break-all whitespace-pre-wrap">
								{JSON.stringify(job.result, null, 2)}
							</pre>
						) : (
							<p className="text-sm text-muted-foreground">no result yet.</p>
						)}
					</div>
				</div>
			)}
		</>
	);
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
	return (
		<>
			<dt className="text-muted-foreground">{label}</dt>
			<dd className="min-w-0 break-words">{children}</dd>
		</>
	);
}
