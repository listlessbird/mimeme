import { AdminSectionError } from "@/components/admin/admin-section-error";
import { SearchEvalRuns } from "@/components/admin/search-eval-overview";
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/admin/evals/runs")({
	errorComponent: AdminSectionError,
	component: SearchEvalRuns,
});
