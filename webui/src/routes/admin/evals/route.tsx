import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { createFileRoute, Link, Outlet, useLocation } from "@tanstack/react-router";

export const Route = createFileRoute("/admin/evals")({
	component: SearchEvalsLayout,
});

function SearchEvalsLayout() {
	const pathname = useLocation({ select: (location) => location.pathname });
	const active = pathname.endsWith("/judge")
		? "judge"
		: pathname.endsWith("/compare")
			? "compare"
			: "overview";

	return (
		<div className="flex flex-col gap-6">
			<header className="flex flex-col gap-4 border-b pb-4 md:flex-row md:items-end md:justify-between">
				<div className="flex max-w-2xl flex-col gap-1">
					<h1 className="text-xl font-semibold tracking-tight">Core search</h1>
					<p className="text-sm text-muted-foreground">
						Judge what users would want, then compare the rankings they actually receive.
					</p>
				</div>
				<Tabs value={active}>
					<TabsList variant="line" aria-label="Core search sections">
						<TabsTrigger value="overview" nativeButton={false} render={<Link to="/admin/evals" />}>
							Overview
						</TabsTrigger>
						<TabsTrigger
							value="judge"
							nativeButton={false}
							render={<Link to="/admin/evals/judge" />}
						>
							Judge
						</TabsTrigger>
						<TabsTrigger
							value="compare"
							nativeButton={false}
							render={<Link to="/admin/evals/compare" />}
						>
							Compare
						</TabsTrigger>
					</TabsList>
				</Tabs>
			</header>
			<Outlet />
		</div>
	);
}
