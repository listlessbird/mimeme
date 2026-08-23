import { AdminSectionError } from "@/components/admin/admin-section-error";
import { SearchEvalComparison } from "@/components/admin/search-eval-comparison";
import { createFileRoute } from "@tanstack/react-router";
import { z } from "zod";

export const Route = createFileRoute("/admin/evals/compare")({
	validateSearch: z.object({
		baseline: z.string().optional(),
		candidate: z.string().optional(),
	}),
	errorComponent: AdminSectionError,
	component: ComparePage,
});

function ComparePage() {
	const search = Route.useSearch();
	return <SearchEvalComparison baselineRunId={search.baseline} candidateRunId={search.candidate} />;
}
