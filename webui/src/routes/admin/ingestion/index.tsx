import { AdminSectionError } from "@/components/admin/admin-section-error";
import { IngestionBrowse } from "@/components/admin/ingestion-browse";
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/admin/ingestion/")({
	errorComponent: AdminSectionError,
	component: IngestionPage,
});

function IngestionPage() {
	return (
		<div className="flex flex-col gap-6">
			<h1 className="text-lg font-semibold">ingestion</h1>
			<IngestionBrowse />
		</div>
	);
}
