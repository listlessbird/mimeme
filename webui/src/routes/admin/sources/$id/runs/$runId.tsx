import { AdminSectionError } from "@/components/admin/admin-section-error";
import { RunItemsGallery } from "@/components/admin/run-items-gallery";
import { createFileRoute, Link } from "@tanstack/react-router";
import { ChevronLeft } from "lucide-react";

export const Route = createFileRoute("/admin/sources/$id/runs/$runId")({
	errorComponent: AdminSectionError,
	component: RunDetailPage,
});

function RunDetailPage() {
	const { id, runId } = Route.useParams();

	return (
		<div className="flex flex-col gap-6">
			<div className="flex flex-col gap-2">
				<Link
					to="/admin/sources/$id"
					params={{ id }}
					className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
				>
					<ChevronLeft className="size-4" />
					back to source
				</Link>
				<h1 className="text-lg font-semibold">run #{runId} — ingest attempts</h1>
				<p className="text-sm text-muted-foreground">
					each attempt's dedup outcome and the canonical image it resolved to.
				</p>
			</div>

			<RunItemsGallery sourceId={Number(id)} runId={Number(runId)} />
		</div>
	);
}
