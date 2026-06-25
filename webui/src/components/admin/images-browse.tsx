import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select";
import { Skeleton } from "@/components/ui/skeleton";
import {
	type ImageSort,
	IMAGES_PAGE_SIZE,
	IMAGE_STATUSES,
	imagesQueryOptions,
	isImageSort,
	isImageStatus,
} from "@/lib/admin/api";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { parseAsInteger, parseAsString, parseAsStringEnum, useQueryStates } from "nuqs";

import { ImageStatusBadge } from "./badges";
import { ItemThumbnail } from "./item-thumbnail";
import { PaginationBar } from "./pagination-bar";

export function ImagesBrowse() {
	const [filters, setFilters] = useQueryStates({
		status: parseAsString.withDefault(""),
		dataset: parseAsString.withDefault(""),
		sort: parseAsStringEnum<ImageSort>(["newest", "oldest"]).withDefault("newest"),
		offset: parseAsInteger.withDefault(0),
	});

	const { data, isPending, isFetching } = useQuery({
		...imagesQueryOptions({
			limit: IMAGES_PAGE_SIZE,
			offset: filters.offset,
			status: isImageStatus(filters.status) ? filters.status : null,
			dataset: filters.dataset || null,
			sort: filters.sort,
		}),
		placeholderData: keepPreviousData,
	});

	return (
		<div className="flex flex-col gap-4">
			<div className="flex flex-wrap items-end gap-3">
				<Field label="status">
					<NativeSelect
						value={filters.status}
						onChange={(e) => setFilters({ status: e.target.value, offset: 0 })}
					>
						<NativeSelectOption value="">all</NativeSelectOption>
						{IMAGE_STATUSES.map((s) => (
							<NativeSelectOption key={s} value={s}>
								{s}
							</NativeSelectOption>
						))}
					</NativeSelect>
				</Field>
				<Field label="dataset">
					<Input
						value={filters.dataset}
						placeholder="any"
						className="h-9 w-44"
						onChange={(e) => setFilters({ dataset: e.target.value, offset: 0 })}
					/>
				</Field>
				<Field label="sort">
					<NativeSelect
						value={filters.sort}
						onChange={(e) =>
							setFilters({
								sort: isImageSort(e.target.value) ? e.target.value : "newest",
								offset: 0,
							})
						}
					>
						<NativeSelectOption value="newest">newest</NativeSelectOption>
						<NativeSelectOption value="oldest">oldest</NativeSelectOption>
					</NativeSelect>
				</Field>
			</div>

			{isPending ? (
				<ImagesSkeleton />
			) : !data || data.images.length === 0 ? (
				<Empty className="border">
					<EmptyHeader>
						<EmptyTitle>no images</EmptyTitle>
						<EmptyDescription>nothing matches these filters yet.</EmptyDescription>
					</EmptyHeader>
				</Empty>
			) : (
				<>
					<div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
						{data.images.map((image) => (
							<Link
								key={image.id}
								to="/admin/images/$imageId"
								params={{ imageId: String(image.id) }}
								className="group flex flex-col gap-2 rounded-md border p-2 transition-colors hover:bg-accent/40 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
							>
								<ItemThumbnail src={image.url ?? null} alt={`image ${image.id}`} />
								<div className="flex items-center justify-between gap-2">
									<span className="font-mono text-xs text-muted-foreground">#{image.id}</span>
									<ImageStatusBadge status={image.status} />
								</div>
								{image.dataset ? (
									<span className="truncate text-xs text-muted-foreground">{image.dataset}</span>
								) : null}
							</Link>
						))}
					</div>
					<PaginationBar
						offset={filters.offset}
						limit={IMAGES_PAGE_SIZE}
						total={data.total}
						loaded={data.images.length}
						isFetching={isFetching}
						onOffsetChange={(offset) => setFilters({ offset })}
					/>
				</>
			)}
		</div>
	);
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
	return (
		<div className="flex flex-col gap-1.5">
			<Label className="text-xs text-muted-foreground">{label}</Label>
			{children}
		</div>
	);
}

function ImagesSkeleton() {
	return (
		<div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
			{Array.from({ length: 8 }).map((_, i) => (
				<Skeleton key={i} className="aspect-square w-full" />
			))}
		</div>
	);
}
