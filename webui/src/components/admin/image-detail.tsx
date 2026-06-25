import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Image } from "@/lib/admin/api";
import { absoluteTime } from "@/lib/admin/format";

import { ProcessingStatusBadge } from "./badges";
import { IdChip, RawJsonDrawer } from "./dev-toolkit";
import { ZoomableImage } from "./zoomable-image";

export function ImageDetail({ image }: { image: Image }) {
	return (
		<div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
			<div className="flex flex-col gap-4">
				<Preview url={image.url ?? null} alt={`image ${image.id}`} />
				{image.caption ? (
					<Card>
						<CardHeader>
							<CardTitle className="text-sm">caption</CardTitle>
						</CardHeader>
						<CardContent className="text-sm break-words">{image.caption}</CardContent>
					</Card>
				) : null}
				{image.ocr_text ? (
					<Card>
						<CardHeader>
							<CardTitle className="text-sm">ocr text</CardTitle>
						</CardHeader>
						<CardContent className="text-sm break-words whitespace-pre-wrap">
							{image.ocr_text}
						</CardContent>
					</Card>
				) : null}
			</div>

			<div className="flex flex-col gap-4">
				<div className="flex flex-wrap items-center gap-2">
					<IdChip label="image_id" value={image.id} />
					<RawJsonDrawer data={image} title={`image #${image.id}`} />
				</div>

				<Card>
					<CardHeader>
						<CardTitle className="text-sm">processing</CardTitle>
					</CardHeader>
					<CardContent>
						<dl className="flex flex-col gap-2 text-sm">
							<StatusRow label="ocr" status={image.ocr_status} />
							<StatusRow label="caption" status={image.caption_status} />
							<StatusRow label="embed" status={image.embed_status} />
						</dl>
					</CardContent>
				</Card>

				<Card>
					<CardHeader>
						<CardTitle className="text-sm">metadata</CardTitle>
					</CardHeader>
					<CardContent>
						<dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
							<Row label="dataset">{image.dataset ?? <Muted>none</Muted>}</Row>
							<Row label="dimensions">
								{image.width && image.height ? (
									`${image.width} × ${image.height}`
								) : (
									<Muted>unknown</Muted>
								)}
							</Row>
							<Row label="format">{image.format ?? <Muted>unknown</Muted>}</Row>
							<Row label="created">
								{image.created_at ? absoluteTime(image.created_at) : <Muted>unknown</Muted>}
							</Row>
						</dl>
						<div className="mt-3 flex flex-col gap-2">
							<IdChip label="sha256" value={image.sha256} truncate />
							{image.s3_key ? <IdChip label="s3_key" value={image.s3_key} truncate /> : null}
							{image.phash ? <IdChip label="phash" value={image.phash} truncate /> : null}
						</div>
					</CardContent>
				</Card>

				{image.tags && image.tags.length > 0 ? (
					<Card>
						<CardHeader>
							<CardTitle className="text-sm">tags</CardTitle>
						</CardHeader>
						<CardContent className="flex flex-wrap gap-1.5">
							{image.tags.map((tag) => (
								<Badge key={tag} variant="secondary">
									{tag}
								</Badge>
							))}
						</CardContent>
					</Card>
				) : null}
			</div>
		</div>
	);
}

function Preview({ url, alt }: { url: string | null; alt: string }) {
	return (
		<div className="flex items-center justify-center rounded-md border bg-muted">
			<ZoomableImage
				src={url}
				alt={alt}
				className="max-h-[70vh] w-auto max-w-full rounded-md object-contain"
				fallbackClassName="aspect-video w-full"
			/>
		</div>
	);
}

function StatusRow({ label, status }: { label: string; status: string | null | undefined }) {
	return (
		<div className="flex items-center justify-between gap-3">
			<dt className="text-muted-foreground">{label}</dt>
			<dd>
				<ProcessingStatusBadge status={status} />
			</dd>
		</div>
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
