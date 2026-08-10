import { ErrorState } from "@/components/error-state";
import { MemeGrid } from "@/components/meme-grid";
import { SearchBar } from "@/components/search-bar";
import { type SearchResponse, searchMemesInfiniteQueryOptions } from "@/lib/api";
import { logError, serializeError } from "@/lib/observability";
import { useSuspenseInfiniteQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { Loader2 } from "lucide-react";
import { createStandardSchemaV1, parseAsString } from "nuqs";
import { useCallback, useEffect, useRef, useState } from "react";

const searchParams = {
	q: parseAsString.withDefault(""),
};

const AUTO_LOAD_WINDOW = 200;

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

		await context.queryClient.ensureInfiniteQueryData(searchMemesInfiniteQueryOptions(deps.q));
		return null;
	},
	pendingMs: 200,
	pendingComponent: ResultsPendingComponent,
	errorComponent: ResultsErrorComponent,
	component: ResultsPage,
});

function ResultsPendingComponent() {
	return (
		<div className="min-h-screen bg-background">
			<div className="sticky top-0 z-40 border-b border-border/50 bg-background/80 backdrop-blur-sm">
				<div className="container mx-auto max-w-[1600px] px-4 pt-4 pb-3 md:px-6 md:pt-6">
					<SearchBar live isSearching />
				</div>
			</div>

			<div className="container mx-auto max-w-[1600px] px-4 pt-6 pb-10 md:px-6">
				<div className="grid grid-cols-2 gap-3 sm:grid-cols-3 sm:gap-4 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6">
					{Array.from({ length: 12 }).map((_, i) => (
						<div key={i} className="aspect-[4/5] animate-pulse rounded-lg bg-muted" />
					))}
				</div>
			</div>
		</div>
	);
}

function ResultsPage() {
	const { q } = Route.useSearch();
	const query = (q ?? "").trim();

	if (!query) {
		return (
			<div className="min-h-screen bg-background">
				<div className="sticky top-0 z-40 border-b border-border/50 bg-background/80 backdrop-blur-sm">
					<div className="container mx-auto max-w-[1600px] px-4 pt-4 pb-3 md:px-6 md:pt-6">
						<SearchBar live />
					</div>
				</div>

				<div className="container mx-auto max-w-[1600px] px-4 pt-6 pb-10 md:px-6">
					<div className="py-24 text-center">
						<p className="text-sm text-foreground">Start with a description</p>
						<p className="mt-2 text-xs text-muted-foreground">
							Describe the meme you want to find in the search box above.
						</p>
					</div>
				</div>
			</div>
		);
	}

	return <QueryBackedResults key={query} query={query} />;
}

function QueryBackedResults({ query }: { query: string }) {
	const loadMoreSentinelRef = useRef<HTMLDivElement | null>(null);
	const [autoLoadLimit, setAutoLoadLimit] = useState(AUTO_LOAD_WINDOW);
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
	const reachedAutoLoadLimit = aggregatedData.results.length >= autoLoadLimit;
	const loadMore = useCallback(
		(force = false) => {
			if (
				!hasNextPage ||
				isFetching ||
				(reachedAutoLoadLimit && !force) ||
				(isFetchNextPageError && !force)
			) {
				return;
			}

			void fetchNextPage();
		},
		[fetchNextPage, hasNextPage, isFetchNextPageError, isFetching, reachedAutoLoadLimit],
	);

	useEffect(() => {
		const node = loadMoreSentinelRef.current;

		if (!node || !hasNextPage || isFetching || isFetchNextPageError || reachedAutoLoadLimit) {
			return undefined;
		}

		const observer = new IntersectionObserver(
			(entries) => {
				if (entries[0]?.isIntersecting) {
					loadMore();
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
	}, [hasNextPage, isFetchNextPageError, isFetching, loadMore, reachedAutoLoadLimit]);

	return (
		<div className="min-h-screen bg-background">
			<div className="sticky top-0 z-40 border-b border-border/50 bg-background/80 backdrop-blur-sm">
				<div className="container mx-auto max-w-[1600px] px-4 pt-4 pb-3 md:px-6 md:pt-6">
					<SearchBar live isSearching={isFetching && !isFetchingNextPage} />
					<div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-xs text-muted-foreground">
						<p>
							<span className="text-foreground">{aggregatedData.results.length}</span> results
							loaded for "{firstPage.query}"
						</p>
						<p>{firstPage.search_time_ms.toFixed(0)} ms</p>
					</div>
				</div>
			</div>

			<main
				className="container mx-auto max-w-[1600px] px-4 pt-6 pb-10 md:px-6"
				aria-busy={isFetchingNextPage}
			>
				<MemeGrid data={aggregatedData} />
				{hasNextPage ? (
					<div
						ref={loadMoreSentinelRef}
						className="mt-8 flex min-h-16 flex-col items-center justify-center gap-3"
						aria-live="polite"
					>
						{isFetchNextPageError ? (
							<div className="flex flex-wrap items-center justify-center gap-3 text-xs">
								<p className="text-destructive">
									{error instanceof Error
										? error.message
										: "could not load more results. try again."}
								</p>
								<button
									type="button"
									onClick={() => loadMore(true)}
									className="rounded-md border border-border bg-card px-3 py-2 text-foreground transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
								>
									Try again
								</button>
							</div>
						) : null}
						{isFetchingNextPage ? (
							<div className="inline-flex items-center gap-2 text-xs text-muted-foreground">
								<Loader2 className="size-4 animate-spin" aria-hidden="true" />
								Loading more results
							</div>
						) : null}
						{reachedAutoLoadLimit && !isFetchNextPageError ? (
							<button
								type="button"
								onClick={() => setAutoLoadLimit((limit) => limit + AUTO_LOAD_WINDOW)}
								className="rounded-md border border-border bg-card px-4 py-2 text-sm text-foreground transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
							>
								Load more results
							</button>
						) : null}
					</div>
				) : aggregatedData.results.length > 0 ? (
					<p className="mt-8 text-center text-xs text-muted-foreground">You've reached the end.</p>
				) : null}
			</main>
		</div>
	);
}

function ResultsErrorComponent({ error, reset }: { error: unknown; reset: () => void }) {
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
