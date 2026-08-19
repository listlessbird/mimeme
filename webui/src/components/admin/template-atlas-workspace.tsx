import { AdminSectionError } from "@/components/admin/admin-section-error";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
	Empty,
	EmptyContent,
	EmptyDescription,
	EmptyHeader,
	EmptyTitle,
} from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import {
	adminQueryKeys,
	runTemplateAtlas,
	templateAtlasQueryOptions,
	type TemplateAtlas,
	type TemplateCluster,
} from "@/lib/admin/api";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RotateCw, Sparkles } from "lucide-react";
import { useMemo, useState } from "react";

const RUN_OPTIONS = {
	neighbors: 20,
	similarity_threshold: 0.72,
	min_cluster_size: 3,
} as const;

export function TemplateAtlasWorkspace() {
	const queryClient = useQueryClient();
	const query = useQuery(templateAtlasQueryOptions());
	const [search, setSearch] = useState("");
	const [selectedId, setSelectedId] = useState<string | null>(null);
	const run = useMutation({
		mutationFn: () => runTemplateAtlas({ data: RUN_OPTIONS }),
		onSuccess: (atlas) => {
			queryClient.setQueryData(adminQueryKeys.templateAtlas, atlas);
			setSelectedId(atlas.clusters?.[0]?.id ?? null);
		},
	});

	if (query.isError) {
		return (
			<AdminSectionError
				error={query.error}
				reset={() => void query.refetch()}
				title="atlas unavailable"
			/>
		);
	}

	const atlas = query.data ?? null;
	const filteredClusters = useMemo(() => filterClusters(atlas, search), [atlas, search]);
	const selectedCluster =
		filteredClusters.find((cluster) => cluster.id === selectedId) ?? filteredClusters[0] ?? null;

	return (
		<div className="flex flex-col gap-6">
			<Header atlas={atlas} isRunning={run.isPending} onRun={() => run.mutate()} />

			{run.isError ? (
				<div className="border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
					{run.error instanceof Error ? run.error.message : "the atlas run failed."}
				</div>
			) : null}

			{query.isPending ? (
				<AtlasLoading />
			) : atlas === null ? (
				<Empty className="min-h-72 border">
					<EmptyHeader>
						<EmptyTitle>no atlas snapshot yet</EmptyTitle>
						<EmptyDescription>
							Run the first experiment to cluster completed SigLIP2 image embeddings. Source
							metadata will seed template names where it is available.
						</EmptyDescription>
					</EmptyHeader>
					<EmptyContent>
						<Button onClick={() => run.mutate()} disabled={run.isPending}>
							<Sparkles data-icon="inline-start" />
							run experiment
						</Button>
					</EmptyContent>
				</Empty>
			) : (
				<>
					<AtlasStats atlas={atlas} />
					<div className="flex flex-col gap-4">
						<div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
							<div>
								<h2 className="text-sm font-semibold">discovered groups</h2>
								<p className="text-xs text-muted-foreground">
									Each card is a connected component in the SigLIP2 kNN graph. The medoid is an
									actual meme.
								</p>
							</div>
							<Input
								value={search}
								onChange={(event) => setSearch(event.target.value)}
								placeholder="filter template labels"
								className="sm:w-64"
							/>
						</div>

						{filteredClusters.length === 0 ? (
							<Empty className="border">
								<EmptyHeader>
									<EmptyTitle>no matching groups</EmptyTitle>
									<EmptyDescription>Try a different template label.</EmptyDescription>
								</EmptyHeader>
							</Empty>
						) : (
							<div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(22rem,0.65fr)]">
								<div className="grid content-start gap-4 md:grid-cols-2">
									{filteredClusters.map((cluster) => (
										<ClusterCard
											key={cluster.id}
											cluster={cluster}
											selected={selectedCluster?.id === cluster.id}
											onSelect={() => setSelectedId(cluster.id)}
										/>
									))}
								</div>
								{selectedCluster ? <ClusterDetail cluster={selectedCluster} /> : null}
							</div>
						)}
					</div>
				</>
			)}
		</div>
	);
}

function Header({
	atlas,
	isRunning,
	onRun,
}: {
	atlas: TemplateAtlas | null | undefined;
	isRunning: boolean;
	onRun: () => void;
}) {
	return (
		<div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
			<div className="flex flex-col gap-1">
				<div className="flex items-center gap-2">
					<h1 className="text-lg font-semibold">template atlas</h1>
					<Badge variant="outline">siglip2 experiment</Badge>
				</div>
				<p className="max-w-2xl text-sm text-muted-foreground">
					A working map of meme-template candidates from existing image embeddings, nearest-neighbor
					edges, and source anchors.
				</p>
				<p className="font-mono text-xs text-muted-foreground">
					k={RUN_OPTIONS.neighbors} · cosine ≥ {RUN_OPTIONS.similarity_threshold} · min group{" "}
					{RUN_OPTIONS.min_cluster_size}
				</p>
			</div>
			<Button variant="outline" onClick={onRun} disabled={isRunning}>
				<RotateCw className={isRunning ? "animate-spin" : undefined} data-icon="inline-start" />
				{isRunning ? "running…" : atlas ? "rerun experiment" : "run experiment"}
			</Button>
		</div>
	);
}

function AtlasStats({ atlas }: { atlas: TemplateAtlas }) {
	const stats = [
		["clusters", atlas.cluster_count.toLocaleString()],
		["clustered images", atlas.clustered_image_count.toLocaleString()],
		["noise", atlas.noise_image_count.toLocaleString()],
		["metadata anchors", atlas.anchor_count.toLocaleString()],
		["graph edges", atlas.graph_edge_count.toLocaleString()],
	];
	return (
		<div className="grid grid-cols-2 gap-px border bg-border sm:grid-cols-5">
			{stats.map(([label, value]) => (
				<div key={label} className="bg-card px-4 py-3">
					<div className="font-mono text-xl font-semibold tabular-nums">{value}</div>
					<div className="text-xs text-muted-foreground">{label}</div>
				</div>
			))}
		</div>
	);
}

function ClusterCard({
	cluster,
	selected,
	onSelect,
}: {
	cluster: TemplateCluster;
	selected: boolean;
	onSelect: () => void;
}) {
	return (
		<button
			type="button"
			onClick={onSelect}
			className={`group flex min-w-0 flex-col gap-3 border bg-card p-3 text-left transition-[border-color,background-color,transform] duration-150 hover:-translate-y-0.5 hover:bg-accent/30 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none ${selected ? "border-foreground" : "border-border"}`}
		>
			<ImageFrame image={cluster.medoid} alt={`${cluster.label} medoid`} className="aspect-[4/3]" />
			<div className="flex items-start justify-between gap-3">
				<div className="min-w-0">
					<div className="truncate text-sm font-semibold">{cluster.label}</div>
					<div className="font-mono text-xs text-muted-foreground">
						{cluster.size.toLocaleString()} variants
					</div>
				</div>
				{(cluster.anchors ?? []).length > 0 ? (
					<Badge variant="secondary">anchored</Badge>
				) : (
					<Badge variant="outline">unlabeled</Badge>
				)}
			</div>
			<div className="grid grid-cols-4 gap-1 overflow-hidden">
				{(cluster.samples ?? []).slice(0, 4).map((image) => (
					<ImageFrame
						key={image.id}
						image={image}
						alt={`variant ${image.id}`}
						className="aspect-square"
						compact
					/>
				))}
			</div>
		</button>
	);
}

function ClusterDetail({ cluster }: { cluster: TemplateCluster }) {
	return (
		<Card className="h-fit xl:sticky xl:top-6">
			<CardHeader>
				<CardTitle className="flex items-center justify-between gap-3 text-base">
					<span className="truncate">{cluster.label}</span>
					<span className="font-mono text-xs font-normal text-muted-foreground">{cluster.id}</span>
				</CardTitle>
				<CardDescription>
					{cluster.size.toLocaleString()} variants · medoid #{cluster.medoid.id}
				</CardDescription>
			</CardHeader>
			<CardContent className="flex flex-col gap-5">
				<div className="grid grid-cols-3 gap-2">
					{(cluster.samples ?? []).map((image) => (
						<a
							key={image.id}
							href={`/admin/images/${image.id}`}
							className="group block focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
						>
							<ImageFrame
								image={image}
								alt={`meme ${image.id}`}
								className="aspect-square"
								compact
							/>
							<div className="mt-1 truncate font-mono text-[10px] text-muted-foreground group-hover:text-foreground">
								#{image.id}
							</div>
						</a>
					))}
				</div>
				{(cluster.anchors ?? []).length > 0 ? (
					<div className="flex flex-col gap-2 border-t pt-4">
						<div className="text-xs font-semibold tracking-[0.14em] text-muted-foreground uppercase">
							source anchors
						</div>
						{(cluster.anchors ?? []).map((anchor) => (
							<div
								key={`${anchor.source_item_id}-${anchor.label}`}
								className="flex items-center justify-between gap-3 text-xs"
							>
								<span className="truncate">{anchor.label}</span>
								<span className="shrink-0 font-mono text-muted-foreground">
									{anchor.image_count} imgs
								</span>
							</div>
						))}
					</div>
				) : null}
			</CardContent>
		</Card>
	);
}

function ImageFrame({
	image,
	alt,
	className,
	compact = false,
}: {
	image: TemplateCluster["medoid"];
	alt: string;
	className: string;
	compact?: boolean;
}) {
	return (
		<div className={`overflow-hidden bg-muted ${className}`}>
			{image.url ? (
				<img
					src={image.url}
					alt={alt}
					loading="lazy"
					className={`size-full object-cover outline outline-1 outline-black/10 ${compact ? "opacity-90 transition-opacity group-hover:opacity-100" : ""}`}
				/>
			) : (
				<div className="flex size-full items-center justify-center font-mono text-xs text-muted-foreground">
					#{image.id}
				</div>
			)}
		</div>
	);
}

function AtlasLoading() {
	return <div className="h-96 animate-pulse border bg-card" aria-label="loading template atlas" />;
}

function filterClusters(atlas: TemplateAtlas | null | undefined, query: string) {
	if (!atlas) return [];
	const normalized = query.trim().toLowerCase();
	if (!normalized) return atlas.clusters ?? [];
	return (atlas.clusters ?? []).filter((cluster) =>
		[cluster.label, ...(cluster.anchors ?? []).map((anchor) => anchor.label)].some((value) =>
			value.toLowerCase().includes(normalized),
		),
	);
}
