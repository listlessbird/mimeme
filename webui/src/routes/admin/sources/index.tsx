import { SourceFormSheet } from "@/components/admin/source-form-sheet";
import { SourcesTable } from "@/components/admin/sources-table";
import { Button } from "@/components/ui/button";
import {
	Empty,
	EmptyContent,
	EmptyDescription,
	EmptyHeader,
	EmptyTitle,
} from "@/components/ui/empty";
import { Skeleton } from "@/components/ui/skeleton";
import { sourcesQueryOptions } from "@/lib/admin/api";
import { useSuspenseQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { Plus } from "lucide-react";
import { useState } from "react";

export const Route = createFileRoute("/admin/sources/")({
	loader: ({ context }) => context.queryClient.ensureQueryData(sourcesQueryOptions()),
	pendingComponent: SourcesPending,
	component: SourcesPage,
});

function PageHeader({ onNew }: { onNew: () => void }) {
	return (
		<div className="flex items-center justify-between">
			<div className="flex flex-col gap-1">
				<h1 className="text-lg font-semibold">sources</h1>
				<p className="text-sm text-muted-foreground">
					external origins polled into the ingest pipeline.
				</p>
			</div>
			<Button onClick={onNew}>
				<Plus data-icon="inline-start" />
				new source
			</Button>
		</div>
	);
}

function SourcesPage() {
	const { data } = useSuspenseQuery(sourcesQueryOptions());
	const [createOpen, setCreateOpen] = useState(false);

	return (
		<div className="flex flex-col gap-6">
			<PageHeader onNew={() => setCreateOpen(true)} />

			{data.sources.length === 0 ? (
				<Empty className="border">
					<EmptyHeader>
						<EmptyTitle>no sources yet</EmptyTitle>
						<EmptyDescription>
							register a source to start acquiring memes on a schedule.
						</EmptyDescription>
					</EmptyHeader>
					<EmptyContent>
						<Button onClick={() => setCreateOpen(true)}>
							<Plus data-icon="inline-start" />
							new source
						</Button>
					</EmptyContent>
				</Empty>
			) : (
				<SourcesTable sources={data.sources} />
			)}

			<SourceFormSheet mode="create" open={createOpen} onOpenChange={setCreateOpen} />
		</div>
	);
}

function SourcesPending() {
	return (
		<div className="flex flex-col gap-6">
			<div className="flex items-center justify-between">
				<Skeleton className="h-9 w-40" />
				<Skeleton className="h-9 w-32" />
			</div>
			<Skeleton className="h-64 w-full" />
		</div>
	);
}
