import { Button } from "@/components/ui/button";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Skeleton } from "@/components/ui/skeleton";
import { Spinner } from "@/components/ui/spinner";
import { ITEMS_PAGE_SIZE, type RunItem, runItemsQueryOptions } from "@/lib/admin/api";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ArrowUpRight, RotateCcw } from "lucide-react";
import { useState } from "react";

import { DedupReasonBadge, IngestStatusBadge } from "./badges";
import { RawJsonDrawer } from "./dev-toolkit";
import { ItemThumbnail } from "./item-thumbnail";
import { PaginationBar } from "./pagination-bar";
import { ImageIdChip } from "./provenance-chips";
import { useRetryItem, useRetryRun } from "./use-source-retry";

export function RunItemsGallery({ sourceId, runId }: { sourceId: number; runId: number }) {
	const [offset, setOffset] = useState(0);
	const { data, isPending, isFetching } = useQuery({
		...runItemsQueryOptions(sourceId, runId, offset),
		placeholderData: keepPreviousData,
	});

	const retryRun = useRetryRun(sourceId, runId);
	const retryItem = useRetryItem(sourceId, runId);

	if (isPending) return <ItemsSkeleton />;

	if (!data || data.items.length === 0) {
		return (
			<Empty className="border">
				<EmptyHeader>
					<EmptyTitle>no ingest attempts for this run</EmptyTitle>
					<EmptyDescription>Empty.</EmptyDescription>
				</EmptyHeader>
			</Empty>
		);
	}

	const hasFailures = data.items.some((item) => item.status === "FAILED");

	return (
		<div className="flex flex-col gap-4">
			{hasFailures ? (
				<div className="flex items-center justify-end">
					<Button
						variant="outline"
						size="sm"
						onClick={() => retryRun.mutate()}
						disabled={retryRun.isPending}
					>
						{retryRun.isPending ? (
							<Spinner data-icon="inline-start" />
						) : (
							<RotateCcw data-icon="inline-start" />
						)}
						retry all failed
					</Button>
				</div>
			) : null}

			<div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
				{data.items.map((item) => (
					<AttemptCard
						key={item.id}
						item={item}
						onRetry={
							item.source_item_id != null
								? () => retryItem.mutate(item.source_item_id as number)
								: undefined
						}
						retrying={retryItem.isPending && retryItem.variables === item.source_item_id}
					/>
				))}
			</div>
			<PaginationBar
				offset={offset}
				limit={ITEMS_PAGE_SIZE}
				total={data.total}
				loaded={data.items.length}
				isFetching={isFetching}
				onOffsetChange={setOffset}
			/>
		</div>
	);
}

function AttemptCard({
	item,
	onRetry,
	retrying,
}: {
	item: RunItem;
	onRetry?: () => void;
	retrying: boolean;
}) {
	return (
		<div className="flex gap-3 rounded-md border p-3">
			<ItemThumbnail
				src={item.thumbnail_url}
				alt={item.title ?? item.external_item_id ?? item.url}
				className="size-20 shrink-0"
			/>
			<div className="flex min-w-0 flex-col gap-1.5">
				<p className="truncate text-sm font-medium" title={item.title ?? undefined}>
					{item.title ?? item.external_item_id ?? "untitled"}
				</p>
				<div className="flex flex-wrap items-center gap-1.5">
					<IngestStatusBadge status={item.status} />
					{item.duplicate_reason ? <DedupReasonBadge reason={item.duplicate_reason} /> : null}
				</div>
				<div className="flex flex-wrap items-center gap-1.5">
					{item.image_id != null ? <ImageIdChip imageId={item.image_id} /> : null}
					<RawJsonDrawer data={item} title={`attempt #${item.id}`} />
				</div>
				{item.duplicate_reason && item.image_id != null && item.thumbnail_url ? (
					<a
						href={item.thumbnail_url}
						target="_blank"
						rel="noreferrer"
						className="inline-flex w-fit items-center gap-0.5 text-xs hover:underline"
					>
						canonical image #{item.image_id}
						<ArrowUpRight className="size-3" />
					</a>
				) : null}
				{item.status === "FAILED" && item.error_message ? (
					<p className="line-clamp-3 text-xs text-destructive" title={item.error_message}>
						{item.error_message}
					</p>
				) : null}
				{item.status === "FAILED" && onRetry ? (
					<Button
						variant="outline"
						size="xs"
						className="mt-0.5 w-fit"
						onClick={onRetry}
						disabled={retrying}
					>
						{retrying ? (
							<Spinner data-icon="inline-start" />
						) : (
							<RotateCcw data-icon="inline-start" />
						)}
						retry
					</Button>
				) : null}
			</div>
		</div>
	);
}

function ItemsSkeleton() {
	return (
		<div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
			{Array.from({ length: 6 }).map((_, i) => (
				<Skeleton key={i} className="h-28 w-full" />
			))}
		</div>
	);
}
