import { GifAnnotationWorkspace } from "@/components/admin/gif-annotation-workspace";
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/admin/gif-annotation/")({
	component: GifAnnotationPage,
});

function GifAnnotationPage() {
	return <GifAnnotationWorkspace />;
}
