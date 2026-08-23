import { AdminSectionError } from "@/components/admin/admin-section-error";
import { SearchEvalQueries } from "@/components/admin/search-eval-overview";
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/admin/evals/queries")({
	errorComponent: AdminSectionError,
	component: SearchEvalQueries,
});
