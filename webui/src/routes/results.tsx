import { useSuspenseInfiniteQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { Loader2 } from "lucide-react";
import { createStandardSchemaV1, parseAsString } from "nuqs";
import { useEffect, useRef, useState } from "react";
import { ErrorState } from "@/components/error-state";
import { MemeGrid } from "@/components/meme-grid";
import { SearchBar } from "@/components/search-bar";
import {
	type SearchResponse,
	searchMemesInfiniteQueryOptions,
} from "@/lib/api";
import { logError, serializeError } from "@/lib/observability";

const searchParams = {
	q: parseAsString.withDefault(""),
};

export const Route = createFileRoute("/results")({
	validateSearch: createStandardSchemaV1(searchParams, {
		partialOutput: true,
	}),
	loaderDeps: ({ search }) => ({
		q: (search.q ?? "").trim(),
	}),
	loader: async ({ deps, context }) => {
		if (!deps.q) {
			return null;
		}

		await context.queryClient.ensureInfiniteQueryData(
			searchMemesInfiniteQueryOptions(deps.q),
		);
	},
	errorComponent: ResultsErrorComponent,
	component: ResultsPage,
});

function ResultsPage() {
	const { q } = Route.useSearch();
	const query = (q ?? "").trim();

	if (!query) {
		return (
			<div className="min-h-screen bg-background">
				<div className="sticky top-0 z-40 border-b border-border/50 bg-background/80 backdrop-blur-sm">
					<div className="mx-auto max-w-6xl px-4 pb-3 pt-4 md:px-6 md:pt-6">
						<SearchBar live />
					</div>
				</div>

				<div className="mx-auto max-w-6xl px-4 pb-6 pt-4 md:px-6">
					<div className="py-20 text-center text-sm text-muted-foreground">
						<p>enter a search query above</p>
					</div>
				</div>
			</div>
		);
	}

	return <QueryBackedResults key={query} query={query} />;
}

function QueryBackedResults({ query }: { query: string }) {
	const [showLoadMore, setShowLoadMore] = useState(false);
	const loadMoreSentinelRef = useRef<HTMLDivElement | null>(null);
	const {
		data,
		error,
		fetchNextPage,
		hasNextPage,
		isFetching,
		isFetchingNextPage,
		isFetchNextPageError,
	} = useSuspenseInfiniteQuery(searchMemesInfiniteQueryOptions(query));
	const firstPage = data.pages[0];
	const aggregatedData: SearchResponse = {
		...firstPage,
		results: data.pages.flatMap((page) => page.results),
	};

	useEffect(() => {
		const node = loadMoreSentinelRef.current;

		if (!node || !hasNextPage || showLoadMore) {
			return;
		}

		const observer = new IntersectionObserver(
			(entries) => {
				if (entries[0]?.isIntersecting) {
					setShowLoadMore(true);
				}
			},
			{
				root: null,
				rootMargin: "120px 0px",
				threshold: 0,
			},
		);

		observer.observe(node);

		return () => {
			observer.disconnect();
		};
	}, [hasNextPage, showLoadMore]);

	return (
		<div className="min-h-screen bg-background">
			<div className="sticky top-0 z-40 border-b border-border/50 bg-background/80 backdrop-blur-sm">
				<div className="mx-auto max-w-6xl px-4 pb-3 pt-4 md:px-6 md:pt-6">
					<SearchBar live isSearching={isFetching && !isFetchingNextPage} />
					<div className="text-xs text-muted-foreground">
						{firstPage.total} results for "{firstPage.query}" (
						{firstPage.search_time_ms.toFixed(0)}ms)
					</div>
				</div>
			</div>

			<div className="mx-auto max-w-6xl px-4 pb-6 pt-4 md:px-6">
				<MemeGrid data={aggregatedData} />
				{hasNextPage ? (
					<div
						ref={loadMoreSentinelRef}
						className="mt-6 flex min-h-12 flex-col items-center justify-center gap-3"
					>
						{isFetchNextPageError ? (
							<p className="text-xs text-destructive">
								{error instanceof Error
									? error.message
									: "could not load more results. try again."}
							</p>
						) : null}
						{showLoadMore ? (
							<button
								type="button"
								onClick={() => {
									if (!isFetchingNextPage) {
										void fetchNextPage();
									}
								}}
								disabled={isFetchingNextPage}
								className="inline-flex items-center gap-2 rounded-md border border-border bg-card px-4 py-2 text-sm text-foreground transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
							>
								{isFetchingNextPage ? (
									<Loader2 className="h-4 w-4 animate-spin" />
								) : null}
								{isFetchingNextPage
									? "loading more"
									: `load ${Math.min(
										firstPage.limit,
										firstPage.total - aggregatedData.results.length,
									)} more`}
							</button>
						) : null}
					</div>
				) : null}
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
			detail="the request did not complete. retry the request."
			onRetry={reset}
		/>
	);
}
