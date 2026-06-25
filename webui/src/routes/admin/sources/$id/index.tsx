import { EnabledBadge } from "@/components/admin/badges";
import { IdChip, RawJsonDrawer } from "@/components/admin/dev-toolkit";
import { RunsTable } from "@/components/admin/runs-table";
import { SourceActions } from "@/components/admin/source-actions";
import { SourceItemsGallery } from "@/components/admin/source-items-gallery";
import { SourceOverview } from "@/components/admin/source-overview";
import { ErrorState } from "@/components/error-state";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { sourceQueryOptions } from "@/lib/admin/api";
import { useSuspenseQuery } from "@tanstack/react-query";
import { createFileRoute, Link } from "@tanstack/react-router";
import { ChevronLeft } from "lucide-react";

export const Route = createFileRoute("/admin/sources/$id/")({
	loader: ({ context, params }) =>
		context.queryClient.ensureQueryData(sourceQueryOptions(Number(params.id))),
	pendingComponent: DetailPending,
	errorComponent: DetailError,
	component: SourceDetailPage,
});

function BackLink() {
	return (
		<Link
			to="/admin/sources"
			className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
		>
			<ChevronLeft className="size-4" />
			all sources
		</Link>
	);
}

function SourceDetailPage() {
	const { id } = Route.useParams();
	const { data: source } = useSuspenseQuery(sourceQueryOptions(Number(id)));

	return (
		<div className="flex flex-col gap-6">
			<div className="flex flex-col gap-2">
				<BackLink />
				<div className="flex flex-wrap items-center gap-3">
					<h1 className="text-lg font-semibold">{source.name}</h1>
					<EnabledBadge enabled={source.enabled} />
					<IdChip label="source_id" value={source.id} />
					<RawJsonDrawer data={source} title={`source #${source.id}`} />
				</div>
			</div>

			<SourceActions source={source} />

			<Tabs defaultValue="overview">
				<TabsList>
					<TabsTrigger value="overview">overview</TabsTrigger>
					<TabsTrigger value="runs">runs</TabsTrigger>
					<TabsTrigger value="items">items</TabsTrigger>
				</TabsList>
				<TabsContent value="overview">
					<SourceOverview source={source} />
				</TabsContent>
				<TabsContent value="runs">
					<RunsTable sourceId={source.id} runs={source.recent_runs} />
				</TabsContent>
				<TabsContent value="items">
					<SourceItemsGallery sourceId={source.id} />
				</TabsContent>
			</Tabs>
		</div>
	);
}

function DetailPending() {
	return (
		<div className="flex flex-col gap-6">
			<Skeleton className="h-8 w-48" />
			<Skeleton className="h-9 w-72" />
			<Skeleton className="h-64 w-full" />
		</div>
	);
}

function DetailError({ reset }: { reset: () => void }) {
	return (
		<ErrorState
			title="could not load source"
			detail="it may have been deleted, or the request failed."
			onRetry={reset}
		/>
	);
}
