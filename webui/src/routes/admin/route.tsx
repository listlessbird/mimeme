import { AdminShell } from "@/components/admin/admin-shell";
import { requireAdminAccess } from "@/lib/admin/guard";
import { createFileRoute, Outlet } from "@tanstack/react-router";

export const Route = createFileRoute("/admin")({
	beforeLoad: async () => {
		await requireAdminAccess();
	},
	component: AdminLayout,
});

function AdminLayout() {
	return (
		<AdminShell>
			<Outlet />
		</AdminShell>
	);
}
