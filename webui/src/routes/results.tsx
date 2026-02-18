import { createFileRoute } from "@tanstack/react-router";
import { createStandardSchemaV1, parseAsInteger, parseAsString } from "nuqs";
import { ErrorState } from "@/components/error-state";
import { MemeGrid } from "@/components/meme-grid";
import { SearchBar } from "@/components/search-bar";
import { searchMemes } from "@/lib/api";
import { logError, serializeError } from "@/lib/observability";

const searchParams = {
	q: parseAsString.withDefault(""),
	limit: parseAsInteger,
	offset: parseAsInteger,
};

export const Route = createFileRoute("/results")({
	validateSearch: createStandardSchemaV1(searchParams, {
		partialOutput: true,
	}),
	loaderDeps: ({ search }) => ({ search }),
	loader: ({ deps }) => {
		if (!deps.search.q) return null;
		return searchMemes({
			data: {
				q: deps.search.q,
				limit: deps.search.limit ?? undefined,
				offset: deps.search.offset ?? undefined,
			},
		});
	},
	errorComponent: ResultsErrorComponent,
	component: ResultsPage,
});

function ResultsPage() {
	const data = Route.useLoaderData();

	return (
		<div className="min-h-screen bg-background p-4 md:p-6">
			<div className="max-w-6xl mx-auto">
				<SearchBar live />
				{data ? (
					<MemeGrid data={data} />
				) : (
					<div className="text-center py-20 text-muted-foreground text-sm">
						<p>enter a search query above</p>
					</div>
				)}
			</div>
		</div>
	);
}

function ResultsErrorComponent({
	error,
	reset,
}: {
	error: unknown;
	reset: () => void;
}) {
	logError("results.loader.error", {
		route: "/results",
		outcome: "error",
		error: serializeError(error),
	});

	return (
		<ErrorState
			title="search failed"
			detail="the meme search request did not complete. retry the request."
			onRetry={reset}
		/>
	);
}
