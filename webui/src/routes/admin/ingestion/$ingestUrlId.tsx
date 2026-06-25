import { IngestionDetailView } from "@/components/admin/ingestion-detail";
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/admin/ingestion/$ingestUrlId")({
	component: IngestionDetailPage,
});

function IngestionDetailPage() {
	const { ingestUrlId } = Route.useParams();
	return <IngestionDetailView ingestUrlId={Number(ingestUrlId)} />;
}
