import { ImageIngestDialog } from "@/components/admin/image-ingest-dialog";
import { ImagesBrowse } from "@/components/admin/images-browse";
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/admin/images/")({
	component: ImagesPage,
});

function ImagesPage() {
	return (
		<div className="flex flex-col gap-6">
			<div className="flex items-center justify-between gap-4">
				<div className="flex flex-col gap-1">
					<h1 className="text-lg font-semibold">images</h1>
					<p className="text-sm text-muted-foreground">
						the catalog — everything the pipeline has stored.
					</p>
				</div>
				<ImageIngestDialog />
			</div>
			<ImagesBrowse />
		</div>
	);
}
