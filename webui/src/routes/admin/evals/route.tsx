import { Button } from "@/components/ui/button";
import { createFileRoute, Link, Outlet, useLocation } from "@tanstack/react-router";

const EVAL_NAVIGATION = [
	{ label: "Overview", to: "/admin/evals" },
	{ label: "Queries", to: "/admin/evals/queries" },
	{ label: "Judgments", to: "/admin/evals/judge" },
	{ label: "Runs", to: "/admin/evals/runs" },
	{ label: "Compare", to: "/admin/evals/compare" },
] as const;

export const Route = createFileRoute("/admin/evals")({
	component: SearchEvalsLayout,
});

function SearchEvalsLayout() {
	const pathname = useLocation({ select: (location) => location.pathname });

	return (
		<div className="mx-auto flex w-full max-w-[100rem] min-w-0 flex-col gap-6">
			<header className="flex min-w-0 flex-col gap-5">
				<div className="max-w-2xl">
					<h1 className="text-xl font-semibold tracking-tight text-balance sm:text-2xl">
						Core Search Eval
					</h1>
					<p className="mt-1 text-sm text-pretty text-muted-foreground">
						Build the benchmark, evaluate the current search system, and inspect what changed.
					</p>
				</div>
				<nav
					aria-label="Core Search Eval"
					className="-mx-4 scrollbar-none w-[calc(100%+2rem)] min-w-0 overflow-x-auto px-4 md:mx-0 md:w-full md:px-0"
				>
					<div className="flex min-w-max gap-1 rounded-lg bg-muted p-1 md:w-fit">
						{EVAL_NAVIGATION.map((item) => {
							const active =
								item.to === "/admin/evals"
									? pathname === "/admin/evals" || pathname === "/admin/evals/"
									: pathname.startsWith(item.to);
							return (
								<Button
									key={item.to}
									variant={active ? "secondary" : "ghost"}
									size="sm"
									nativeButton={false}
									render={<Link to={item.to} aria-current={active ? "page" : undefined} />}
								>
									{item.label}
								</Button>
							);
						})}
					</div>
				</nav>
			</header>
			<Outlet />
		</div>
	);
}
