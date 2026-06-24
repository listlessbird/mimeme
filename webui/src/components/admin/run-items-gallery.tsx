import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Skeleton } from "@/components/ui/skeleton";
import { ITEMS_PAGE_SIZE, type RunItem, runItemsQueryOptions } from "@/lib/admin/api";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ArrowUpRight } from "lucide-react";
import { useState } from "react";

import { DedupReasonBadge, IngestStatusBadge } from "./badges";
import { ItemThumbnail } from "./item-thumbnail";
import { PaginationBar } from "./pagination-bar";

export function RunItemsGallery({ sourceId, runId }: { sourceId: number; runId: number }) {
	const [offset, setOffset] = useState(0);
	const { data, isPending, isFetching } = useQuery({
		...runItemsQueryOptions(sourceId, runId, offset),
		placeholderData: keepPreviousData,
	});

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

	return (
		<div className="flex flex-col gap-4">
			<div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
				{data.items.map((item) => (
					<AttemptCard key={item.id} item={item} />
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

function AttemptCard({ item }: { item: RunItem }) {
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
			</div>
		</div>
	);
}

function ItemsSkeleton() {
	return (
		<div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
			{Array.from({ length: 6 }).map((_, i) => (
				// oxlint-disable-next-line react/no-array-index-key -- static skeleton placeholders, never reordered
				<Skeleton key={i} className="h-28 w-full" />
			))}
		</div>
	);
}
