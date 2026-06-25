import { Badge } from "@/components/ui/badge";
import type {
	DuplicateReason,
	ImageStatus,
	JobStatus,
	ProcessingStatus,
	SourceItemIngestState,
	SourceRunStatus,
	SourceRunTrigger,
} from "@/lib/admin/api";

type BadgeVariant = "default" | "secondary" | "destructive" | "outline";

const RUN_STATUS_VARIANT: Record<SourceRunStatus, BadgeVariant> = {
	COMPLETED: "default",
	RUNNING: "secondary",
	PENDING: "outline",
	PARTIAL: "secondary",
	FAILED: "destructive",
};

export function RunStatusBadge({ status }: { status: SourceRunStatus }) {
	return <Badge variant={RUN_STATUS_VARIANT[status]}>{status.toLowerCase()}</Badge>;
}

export function TriggerBadge({ trigger }: { trigger: SourceRunTrigger }) {
	return <Badge variant="outline">{trigger}</Badge>;
}

const INGEST_STATUS_VARIANT: Record<ProcessingStatus, BadgeVariant> = {
	DONE: "default",
	RUNNING: "secondary",
	PENDING: "outline",
	FAILED: "destructive",
};

export function IngestStatusBadge({ status }: { status: ProcessingStatus }) {
	return <Badge variant={INGEST_STATUS_VARIANT[status]}>{status.toLowerCase()}</Badge>;
}

function isProcessingStatus(value: string): value is ProcessingStatus {
	return value in INGEST_STATUS_VARIANT;
}

export function ProcessingStatusBadge({ status }: { status: string | null | undefined }) {
	if (!status) return <Badge variant="outline">—</Badge>;
	const upper = status.toUpperCase();
	const variant = isProcessingStatus(upper) ? INGEST_STATUS_VARIANT[upper] : "outline";
	return <Badge variant={variant}>{status.toLowerCase()}</Badge>;
}

export function DedupReasonBadge({ reason }: { reason: DuplicateReason }) {
	return <Badge variant="secondary">deduped · {reason.toLowerCase()}</Badge>;
}

export function EnabledBadge({ enabled }: { enabled: boolean }) {
	return (
		<Badge variant={enabled ? "default" : "outline"}>{enabled ? "enabled" : "disabled"}</Badge>
	);
}

const IMAGE_STATUS_VARIANT: Record<ImageStatus, BadgeVariant> = {
	done: "default",
	embedding: "secondary",
	annotating: "secondary",
	scanning: "secondary",
	downloading: "secondary",
	pending: "outline",
	failed: "destructive",
};

export function ImageStatusBadge({ status }: { status: ImageStatus }) {
	return <Badge variant={IMAGE_STATUS_VARIANT[status]}>{status}</Badge>;
}

const JOB_STATUS_VARIANT: Record<JobStatus, BadgeVariant> = {
	COMPLETED: "default",
	RUNNING: "secondary",
	PENDING: "outline",
	CANCELED: "outline",
	FAILED: "destructive",
};

export function JobStatusBadge({ status }: { status: JobStatus }) {
	return <Badge variant={JOB_STATUS_VARIANT[status]}>{status.toLowerCase()}</Badge>;
}

const INGEST_STATE_META: Record<SourceItemIngestState, { variant: BadgeVariant; label: string }> = {
	ingested: { variant: "default", label: "ingested" },
	deduped: { variant: "secondary", label: "deduped" },
	failed: { variant: "destructive", label: "failed" },
	in_flight: { variant: "secondary", label: "in-flight" },
	unknown: { variant: "outline", label: "unknown" },
};

export function IngestStateBadge({ state }: { state: SourceItemIngestState }) {
	const meta = INGEST_STATE_META[state];
	return <Badge variant={meta.variant}>{meta.label}</Badge>;
}
