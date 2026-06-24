import { Badge } from "@/components/ui/badge";
import type {
	DuplicateReason,
	ProcessingStatus,
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

export function DedupReasonBadge({ reason }: { reason: DuplicateReason }) {
	return <Badge variant="secondary">deduped · {reason.toLowerCase()}</Badge>;
}

export function EnabledBadge({ enabled }: { enabled: boolean }) {
	return (
		<Badge variant={enabled ? "default" : "outline"}>{enabled ? "enabled" : "disabled"}</Badge>
	);
}
