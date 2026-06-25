import { ImageDetail } from "@/components/admin/image-detail";
import { ErrorState } from "@/components/error-state";
import { Skeleton } from "@/components/ui/skeleton";
import { imageQueryOptions } from "@/lib/admin/api";
import { useSuspenseQuery } from "@tanstack/react-query";
import { createFileRoute, Link } from "@tanstack/react-router";
import { ChevronLeft } from "lucide-react";

export const Route = createFileRoute("/admin/images/$imageId")({
	loader: ({ context, params }) =>
		context.queryClient.ensureQueryData(imageQueryOptions(Number(params.imageId))),
	pendingComponent: DetailPending,
	errorComponent: DetailError,
	component: ImageDetailPage,
});

function ImageDetailPage() {
	const { imageId } = Route.useParams();
	const { data: image } = useSuspenseQuery(imageQueryOptions(Number(imageId)));

	return (
		<div className="flex flex-col gap-6">
			<div className="flex flex-col gap-2">
				<Link
					to="/admin/images"
					className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
				>
					<ChevronLeft className="size-4" />
					all images
				</Link>
				<h1 className="font-mono text-lg font-semibold">image #{image.id}</h1>
			</div>
			<ImageDetail image={image} />
		</div>
	);
}

function DetailPending() {
	return (
		<div className="flex flex-col gap-6">
			<Skeleton className="h-8 w-40" />
			<Skeleton className="h-96 w-full" />
		</div>
	);
}

function DetailError({ reset }: { reset: () => void }) {
	return (
		<ErrorState
			title="could not load image"
			detail="it may have been deleted, or the request failed."
			onRetry={reset}
		/>
	);
}
