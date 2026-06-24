import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogHeader,
	DialogTitle,
} from "@/components/ui/dialog";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Skeleton } from "@/components/ui/skeleton";
import { ITEMS_PAGE_SIZE, type SourceItem, sourceItemsQueryOptions } from "@/lib/admin/api";
import { absoluteTime, relativeTime } from "@/lib/admin/format";
import { cn } from "@/lib/utils";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ArrowBigUp, ArrowUpRight, ChevronLeft, ChevronRight, ImageOff } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { ItemThumbnail } from "./item-thumbnail";
import { PaginationBar } from "./pagination-bar";

export function SourceItemsGallery({ sourceId }: { sourceId: number }) {
	const [offset, setOffset] = useState(0);
	const [openIndex, setOpenIndex] = useState<number | null>(null);
	const { data, isPending, isFetching } = useQuery({
		...sourceItemsQueryOptions(sourceId, offset),
		placeholderData: keepPreviousData,
	});

	return (
		<div className="flex flex-col gap-4">
			<Alert>
				<AlertTitle>this is the source's discovery memory</AlertTitle>
				<AlertDescription>
					everything the source has ever discovered lives here. items it had already seen in a prior
					run are skipped before download, so they won't appear under a later run's items — only
					here.
				</AlertDescription>
			</Alert>

			{isPending ? (
				<ItemsSkeleton />
			) : !data || data.items.length === 0 ? (
				<Empty className="border">
					<EmptyHeader>
						<EmptyTitle>no items discovered yet</EmptyTitle>
						<EmptyDescription>
							once a run completes, discovered items show up here.
						</EmptyDescription>
					</EmptyHeader>
				</Empty>
			) : (
				<>
					<div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
						{data.items.map((item, i) => (
							<ItemCard key={item.id} item={item} onOpen={() => setOpenIndex(i)} />
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
					<ItemLightbox items={data.items} index={openIndex} onIndexChange={setOpenIndex} />
				</>
			)}
		</div>
	);
}

function ItemCard({ item, onOpen }: { item: SourceItem; onOpen: () => void }) {
	const meta = item.raw_metadata;

	return (
		<button
			type="button"
			onClick={onOpen}
			className="flex gap-3 rounded-md border p-3 text-left transition-colors hover:bg-accent/40 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
		>
			<ItemThumbnail
				src={item.thumbnail_url}
				alt={item.title ?? item.external_item_id}
				className="size-20 shrink-0"
			/>
			<div className="flex min-w-0 flex-col gap-1">
				<p className="truncate text-sm font-medium" title={item.title ?? undefined}>
					{item.title ?? item.external_item_id}
				</p>
				<div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
					{meta?.subreddit ? <span>r/{meta.subreddit}</span> : null}
					{meta?.author ? <span>{meta.author}</span> : null}
					{typeof meta?.ups === "number" ? (
						<span className="inline-flex items-center gap-0.5">
							<ArrowBigUp className="size-3" />
							{meta.ups}
						</span>
					) : null}
				</div>
				<p className="text-xs text-muted-foreground">last seen {relativeTime(item.last_seen_at)}</p>
				{meta?.postLink ? (
					<span className="inline-flex w-fit items-center gap-0.5 text-xs text-muted-foreground">
						post
						<ArrowUpRight className="size-3" />
					</span>
				) : null}
			</div>
		</button>
	);
}

function ItemLightbox({
	items,
	index,
	onIndexChange,
}: {
	items: SourceItem[];
	index: number | null;
	onIndexChange: (index: number | null) => void;
}) {
	const open = index !== null;
	const item = index !== null ? items[index] : null;
	const hasPrev = index !== null && index > 0;
	const hasNext = index !== null && index < items.length - 1;

	const goPrev = useCallback(() => {
		onIndexChange(index !== null && index > 0 ? index - 1 : index);
	}, [index, onIndexChange]);

	const goNext = useCallback(() => {
		onIndexChange(index !== null && index < items.length - 1 ? index + 1 : index);
	}, [index, items.length, onIndexChange]);

	useEffect(() => {
		if (!open) return undefined;
		function onKey(e: KeyboardEvent) {
			if (e.key === "ArrowLeft") {
				e.preventDefault();
				goPrev();
			} else if (e.key === "ArrowRight") {
				e.preventDefault();
				goNext();
			}
		}
		window.addEventListener("keydown", onKey);
		return () => window.removeEventListener("keydown", onKey);
	}, [open, goPrev, goNext]);

	const meta = item?.raw_metadata;
	const previewSrc = bestPreview(item) ?? item?.thumbnail_url ?? null;

	return (
		<Dialog open={open} onOpenChange={(next) => !next && onIndexChange(null)}>
			<DialogContent className="max-w-2xl gap-4">
				{item ? (
					<>
						<DialogHeader>
							<DialogTitle className="truncate pr-6">
								{item.title ?? item.external_item_id}
							</DialogTitle>
							<DialogDescription>
								{index !== null ? `item ${index + 1} of ${items.length}` : null} · use ← → to browse
							</DialogDescription>
						</DialogHeader>

						<div className="flex items-center justify-center rounded-md border bg-muted">
							<LightboxImage
								key={item.id}
								src={previewSrc}
								alt={item.title ?? item.external_item_id}
							/>
						</div>

						<dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
							<DataRow label="id">{item.id}</DataRow>
							<DataRow label="external id">{item.external_item_id}</DataRow>
							{meta?.subreddit ? <DataRow label="subreddit">r/{meta.subreddit}</DataRow> : null}
							{meta?.author ? <DataRow label="author">{meta.author}</DataRow> : null}
							{typeof meta?.ups === "number" ? (
								<DataRow label="ups">
									<span className="inline-flex items-center gap-0.5">
										<ArrowBigUp className="size-3.5" />
										{meta.ups}
									</span>
								</DataRow>
							) : null}
							<DataRow label="first seen">
								<span title={absoluteTime(item.first_seen_at)}>
									{relativeTime(item.first_seen_at)}
								</span>
							</DataRow>
							<DataRow label="last seen">
								<span title={absoluteTime(item.last_seen_at)}>
									{relativeTime(item.last_seen_at)}
								</span>
							</DataRow>
							{meta?.postLink ? (
								<DataRow label="post">
									<a
										href={meta.postLink}
										target="_blank"
										rel="noreferrer"
										className="inline-flex w-fit items-center gap-0.5 break-all hover:underline"
									>
										{meta.postLink}
										<ArrowUpRight className="size-3.5 shrink-0" />
									</a>
								</DataRow>
							) : null}
							{item.thumbnail_url ? (
								<DataRow label="thumbnail">
									<a
										href={item.thumbnail_url}
										target="_blank"
										rel="noreferrer"
										className="inline-flex w-fit items-center gap-0.5 break-all hover:underline"
									>
										{item.thumbnail_url}
										<ArrowUpRight className="size-3.5 shrink-0" />
									</a>
								</DataRow>
							) : null}
						</dl>

						<div className="flex items-center justify-between gap-2">
							<Button variant="outline" size="sm" onClick={goPrev} disabled={!hasPrev}>
								<ChevronLeft className="size-4" />
								prev
							</Button>
							<Button variant="outline" size="sm" onClick={goNext} disabled={!hasNext}>
								next
								<ChevronRight className="size-4" />
							</Button>
						</div>
					</>
				) : null}
			</DialogContent>
		</Dialog>
	);
}

function DataRow({ label, children }: { label: string; children: React.ReactNode }) {
	return (
		<>
			<dt className="text-muted-foreground">{label}</dt>
			<dd className="min-w-0 break-words">{children}</dd>
		</>
	);
}

function LightboxImage({ src, alt }: { src: string | null; alt: string }) {
	const [failed, setFailed] = useState(false);

	if (!src || failed) {
		return (
			<div
				className={cn("flex aspect-video w-full items-center justify-center text-muted-foreground")}
			>
				<ImageOff className="size-6" />
			</div>
		);
	}

	return (
		<img
			src={src}
			alt={alt}
			onError={() => setFailed(true)}
			className="max-h-[55vh] w-auto max-w-full rounded-md object-contain"
		/>
	);
}

function bestPreview(item: SourceItem | null): string | null {
	const preview = item?.raw_metadata?.preview;
	if (Array.isArray(preview)) return preview[preview.length - 1] ?? null;
	if (typeof preview === "string") return preview;
	return null;
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
