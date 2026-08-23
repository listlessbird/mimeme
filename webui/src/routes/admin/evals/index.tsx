import { AdminSectionError } from "@/components/admin/admin-section-error";
import { SearchEvalOverview } from "@/components/admin/search-eval-overview";
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/admin/evals/")({
	errorComponent: AdminSectionError,
	component: SearchEvalOverview,
});
