import { Button } from "@/components/ui/button";
import { createFileRoute, Link, Outlet, useLocation } from "@tanstack/react-router";
import { ArrowLeft } from "lucide-react";

export const Route = createFileRoute("/admin/evals")({
	component: SearchEvalsLayout,
});

function SearchEvalsLayout() {
	const pathname = useLocation({ select: (location) => location.pathname });
	const isOverview = pathname === "/admin/evals" || pathname === "/admin/evals/";

	return (
		<div className="flex flex-col gap-6">
			<header className="flex items-center justify-between gap-4 border-b pb-4">
				<h1 className="text-xl font-semibold tracking-tight">Search evaluation</h1>
				{isOverview ? null : (
					<Button
						variant="ghost"
						size="sm"
						nativeButton={false}
						render={<Link to="/admin/evals" />}
					>
						<ArrowLeft data-icon="inline-start" /> Back to evaluation
					</Button>
				)}
			</header>
			<Outlet />
		</div>
	);
}
