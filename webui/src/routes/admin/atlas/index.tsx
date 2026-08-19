import { AdminSectionError } from "@/components/admin/admin-section-error";
import { TemplateAtlasWorkspace } from "@/components/admin/template-atlas-workspace";
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/admin/atlas/")({
	errorComponent: AdminSectionError,
	component: TemplateAtlasPage,
});

function TemplateAtlasPage() {
	return <TemplateAtlasWorkspace />;
}
