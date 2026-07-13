import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
	type Image,
	type IngestionDetail,
	INGESTION_POLL_MS,
	imageQueryOptions,
	ingestionAttemptQueryOptions,
} from "@/lib/admin/api";
import { absoluteTime, relativeTime } from "@/lib/admin/format";
import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { AlertTriangle, ChevronLeft } from "lucide-react";

import { IngestOutcomeBadge, StageBadge, TriggerBadge } from "./badges";
import { IdChip, RawJsonDrawer } from "./dev-toolkit";
import { ImageDetail } from "./image-detail";
import { IngestionLogs } from "./ingestion-logs";
import { JobIdChip } from "./job-detail";
import { ImageIdChip } from "./provenance-chips";
import { ZoomableImage } from "./zoomable-image";

const IN_FLIGHT = new Set<IngestionDetail["status"]>(["PENDING", "RUNNING"]);

export function IngestionDetailView({ ingestUrlId }: { ingestUrlId: number }) {
	const { data: attempt, isPending } = useQuery({
		...ingestionAttemptQueryOptions(ingestUrlId),
		refetchInterval: (query) => {
			const status = query.state.data?.status;
			return status !== undefined && IN_FLIGHT.has(status) ? INGESTION_POLL_MS : false;
		},
		throwOnError: true,
	});

	if (isPending) return <DetailSkeleton />;
	if (!attempt)
		return <p className="text-sm text-muted-foreground">could not load this attempt.</p>;

	const failed = attempt.status === "FAILED";
	const inFlight = IN_FLIGHT.has(attempt.status);

	return (
		<div className="flex flex-col gap-6">
			<div className="flex flex-col gap-2">
				<Link
					to="/admin/ingestion"
					className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
				>
					<ChevronLeft className="size-4" />
					all ingestion
				</Link>
				<div className="flex flex-wrap items-center gap-2">
					<h1 className="text-lg font-semibold">attempt #{attempt.ingest_url_id}</h1>
					{inFlight || failed ? <StageBadge stage={attempt.stage} frozen={failed} /> : null}
					<IngestOutcomeBadge outcome={attempt.outcome} />
					<TriggerBadge trigger={attempt.trigger} />
					{attempt.source_id != null && attempt.source_name ? (
						<Link
							to="/admin/sources/$id"
							params={{ id: String(attempt.source_id) }}
							className="text-sm font-medium hover:underline"
						>
							{attempt.source_name}
						</Link>
					) : (
						<span className="text-sm text-muted-foreground">manual upload</span>
					)}
				</div>
			</div>

			<div className="flex flex-wrap items-center gap-2">
				<IdChip label="ingest_url" value={attempt.ingest_url_id} />
				<JobIdChip jobId={attempt.job_id} />
				{attempt.source_run_id != null ? (
					<IdChip label="source_run_id" value={attempt.source_run_id} />
				) : null}
				{attempt.resolved_image_id != null ? (
					<ImageIdChip imageId={attempt.resolved_image_id} />
				) : null}
				<RawJsonDrawer data={attempt} title={`attempt #${attempt.ingest_url_id}`} />
			</div>

			{failed ? (
				<Card className="border-destructive/40">
					<CardHeader>
						<CardTitle className="flex items-center gap-2 text-sm text-destructive">
							<AlertTriangle className="size-4" />
							failed in {attempt.stage.toLowerCase()}
						</CardTitle>
					</CardHeader>
					<CardContent className="text-sm break-words whitespace-pre-wrap">
						{attempt.error_message ?? "no error message recorded."}
					</CardContent>
				</Card>
			) : (
				<Card>
					<CardHeader>
						<CardTitle className="text-sm">pipeline stage</CardTitle>
					</CardHeader>
					<CardContent>
						<dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
							<Row label={inFlight ? "stage" : "outcome"}>
								{inFlight ? (
									<StageBadge stage={attempt.stage} />
								) : (
									<IngestOutcomeBadge outcome={attempt.outcome} />
								)}
							</Row>
							<Row label="status">{attempt.status.toLowerCase()}</Row>
							<Row label={inFlight ? "in stage" : "finished"}>
								{attempt.stage_updated_at ? (
									<span title={absoluteTime(attempt.stage_updated_at)}>
										{relativeTime(attempt.stage_updated_at)}
									</span>
								) : (
									<Muted>—</Muted>
								)}
							</Row>
							<Row label="created">
								<span title={absoluteTime(attempt.created_at)}>
									{relativeTime(attempt.created_at)}
								</span>
							</Row>
						</dl>
					</CardContent>
				</Card>
			)}

			{attempt.outcome === "deduped" ? <DedupCard attempt={attempt} /> : null}

			<IngestionLogs ingestUrlId={attempt.ingest_url_id} poll={inFlight} />

			<StoredImage attempt={attempt} />
		</div>
	);
}

function StoredImage({ attempt }: { attempt: IngestionDetail }) {
	const imageId = attempt.resolved_image_id;
	const { data: image, isPending } = useQuery({
		...imageQueryOptions(imageId ?? 0),
		enabled: imageId != null,
	});

	if (imageId == null) {
		return (
			<Card>
				<CardHeader>
					<CardTitle className="text-sm">stored image</CardTitle>
				</CardHeader>
				<CardContent className="flex items-center gap-3">
					<ZoomableImage
						src={attempt.thumbnail_url ?? null}
						alt={`attempt ${attempt.ingest_url_id}`}
						className="size-16 shrink-0 rounded-md border bg-muted object-cover"
						fallbackClassName="size-16 shrink-0"
					/>
					<p className="text-sm text-muted-foreground">
						no stored image yet — this attempt has not landed in the catalog.
					</p>
				</CardContent>
			</Card>
		);
	}

	if (isPending || !image) {
		return <ImagePending />;
	}

	return (
		<div className="flex flex-col gap-4">
			<h2 className="text-sm font-semibold text-muted-foreground">stored image</h2>
			<ImageDetail image={image} />
			<EmbeddingCard image={image} />
		</div>
	);
}

function EmbeddingCard({ image }: { image: Image }) {
	return (
		<Card>
			<CardHeader>
				<CardTitle className="text-sm">models & embedding</CardTitle>
			</CardHeader>
			<CardContent>
				<dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
					<Row label="ocr model">{image.ocr_model ?? <Muted>—</Muted>}</Row>
					<Row label="caption model">{image.caption_model ?? <Muted>—</Muted>}</Row>
					<Row label="embed model">{image.embed_model ?? <Muted>—</Muted>}</Row>
					<Row label="embed dim">{image.embed_dim ?? <Muted>—</Muted>}</Row>
					<Row label="file size">
						{image.file_size != null ? (
							`${image.file_size.toLocaleString()} bytes`
						) : (
							<Muted>—</Muted>
						)}
					</Row>
				</dl>
				{image.embed_s3_key ? (
					<div className="mt-3">
						<IdChip label="embed_key" value={image.embed_s3_key} truncate />
					</div>
				) : null}
			</CardContent>
		</Card>
	);
}

function DedupCard({ attempt }: { attempt: IngestionDetail }) {
	return (
		<Card>
			<CardHeader>
				<CardTitle className="text-sm">dedup decision</CardTitle>
			</CardHeader>
			<CardContent className="flex flex-col gap-2 text-sm">
				<div className="flex items-center gap-2">
					<span className="text-muted-foreground">collapsed via</span>
					<span className="font-mono">{attempt.duplicate_reason ?? "—"}</span>
				</div>
				{attempt.duplicate_of_image_id != null ? (
					<div className="flex items-center gap-2">
						<span className="text-muted-foreground">canonical image</span>
						<ImageIdChip imageId={attempt.duplicate_of_image_id} />
					</div>
				) : null}
			</CardContent>
		</Card>
	);
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
	return (
		<>
			<dt className="text-muted-foreground">{label}</dt>
			<dd className="min-w-0 break-words">{children}</dd>
		</>
	);
}

function Muted({ children }: { children: React.ReactNode }) {
	return <span className="text-muted-foreground">{children}</span>;
}

function ImagePending() {
	return (
		<Card>
			<CardHeader>
				<CardTitle className="text-sm">stored image</CardTitle>
			</CardHeader>
			<CardContent>
				<div className="aspect-video w-full animate-pulse rounded-md bg-muted" />
			</CardContent>
		</Card>
	);
}

function DetailSkeleton() {
	return (
		<div className="flex flex-col gap-4">
			<div className="h-6 w-48 animate-pulse rounded bg-muted" />
			<div className="h-24 w-full animate-pulse rounded bg-muted" />
			<div className="aspect-video w-full animate-pulse rounded bg-muted" />
		</div>
	);
}
