import { AdminSectionError } from "@/components/admin/admin-section-error";
import { SearchEvalJudgment } from "@/components/admin/search-eval-judgment";
import { createFileRoute } from "@tanstack/react-router";
import { z } from "zod";

export const Route = createFileRoute("/admin/evals/judge")({
	validateSearch: z.object({ query: z.coerce.number().int().positive().optional() }),
	errorComponent: AdminSectionError,
	component: JudgePage,
});

function JudgePage() {
	const { query } = Route.useSearch();
	return <SearchEvalJudgment queryId={query} />;
}
