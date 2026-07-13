import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
import { FilterSelect } from "./filter-select";
import { PaginationBar } from "./pagination-bar";
import { ZoomableImage } from "./zoomable-image";

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
		throwOnError: true,
	});

	return (
		<div className="flex flex-col gap-4">
			<div className="flex flex-wrap items-end gap-3">
				<Field label="status">
					<FilterSelect
						className="h-9 w-44"
						anyLabel="all"
						placeholder="all"
						value={filters.status}
						onValueChange={(v) => setFilters({ status: v, offset: 0 })}
						options={IMAGE_STATUSES.map((s) => ({ value: s, label: s }))}
					/>
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
					<FilterSelect
						className="h-9 w-40"
						includeAny={false}
						value={filters.sort}
						onValueChange={(v) => setFilters({ sort: isImageSort(v) ? v : "newest", offset: 0 })}
						options={[
							{ value: "newest", label: "newest" },
							{ value: "oldest", label: "oldest" },
						]}
					/>
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
							<div
								key={image.id}
								className="flex flex-col gap-2 rounded-md border p-2 transition-colors hover:bg-accent/40"
							>
								<ZoomableImage
									src={image.url ?? null}
									alt={`image ${image.id}`}
									triggerClassName="w-full"
									className="aspect-square w-full rounded-md border bg-muted object-cover"
									fallbackClassName="aspect-square w-full"
								/>
								<Link
									to="/admin/images/$imageId"
									params={{ imageId: String(image.id) }}
									className="flex items-center justify-between gap-2 rounded-sm focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
								>
									<span className="font-mono text-xs text-muted-foreground hover:underline">
										#{image.id}
									</span>
									<ImageStatusBadge status={image.status} />
								</Link>
								{image.dataset ? (
									<span className="truncate text-xs text-muted-foreground">{image.dataset}</span>
								) : null}
							</div>
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
