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
		<div className="min-h-screen bg-background">
			<div className="sticky top-0 z-40 bg-background/80 backdrop-blur-sm border-b border-border/50">
				<div className="max-w-6xl mx-auto px-4 md:px-6 pt-4 md:pt-6 pb-3">
					<SearchBar live />
					{data && data.results.length > 0 && (
						<div className="text-xs text-muted-foreground">
							{data.total} results for "{data.query}" ({data.search_time_ms.toFixed(0)}ms)
						</div>
					)}
				</div>
			</div>

			<div className="max-w-6xl mx-auto px-4 md:px-6 pt-4 pb-6">
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
