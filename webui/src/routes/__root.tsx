import { useEffect } from "react";
import { TanStackDevtools } from "@tanstack/react-devtools";
import type { QueryClient } from "@tanstack/react-query";
import {
	createRootRouteWithContext,
	HeadContent,
	Scripts,
} from "@tanstack/react-router";
import { TanStackRouterDevtoolsPanel } from "@tanstack/react-router-devtools";
import { NuqsAdapter } from "nuqs/adapters/tanstack-router";

import { ErrorState } from "@/components/error-state";
import { logError, logInfo, serializeError } from "@/lib/observability";
import appCss from "@/styles.css?url";
import TanStackQueryDevtools from "../integrations/tanstack-query/devtools";

interface MyRouterContext {
	queryClient: QueryClient;
}

export const Route = createRootRouteWithContext<MyRouterContext>()({
	head: () => ({
		meta: [
			{
				charSet: "utf-8",
			},
			{
				name: "viewport",
				content: "width=device-width, initial-scale=1",
			},
			{
				title: "mìmeme (觅meme)",
			},
		],
		links: [
			{
				rel: "stylesheet",
				href: appCss,
			},
			{
				rel: "icon",
				type: "image/png",
				href: "/favicon-96x96.png",
				sizes: "96x96",
			},
			{
				rel: "icon",
				type: "image/svg+xml",
				href: "/favicon.svg",
			},
			{
				rel: "shortcut icon",
				href: "/favicon.ico",
			},
			{
				rel: "apple-touch-icon",
				sizes: "180x180",
				href: "/apple-touch-icon.png",
			},
			{
				rel: "manifest",
				href: "/site.webmanifest",
			},
		],
	}),
	errorComponent: RootErrorComponent,
	notFoundComponent: RootNotFoundComponent,
	shellComponent: RootDocument,
});

function RootDocument({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    if (import.meta.env.DEV) {
      void import("react-grab");
      void import("@react-grab/mcp/client");
    }
  }, []);


	return (
		<html lang="en">
			<head>
				<HeadContent />
			</head>
			<body className="font-mono antialiased">
				<NuqsAdapter>{children}</NuqsAdapter>
				<TanStackDevtools
					config={{
						position: "bottom-right",
					}}
					plugins={[
						{
							name: "Tanstack Router",
							render: <TanStackRouterDevtoolsPanel />,
						},
						TanStackQueryDevtools,
					]}
				/>
				<Scripts />
			</body>
		</html>
	);
}

function RootErrorComponent({
	error,
	reset,
}: {
	error: unknown;
	reset: () => void;
}) {
	logError("route.root.error", {
		route: "root",
		outcome: "error",
		error: serializeError(error),
	});

	return (
		<RootDocument>
			<ErrorState
				title="something went wrong"
				detail="an unexpected error occurred while rendering this page."
				onRetry={reset}
			/>
		</RootDocument>
	);
}

function RootNotFoundComponent() {
	logInfo("route.root.not_found", {
		route: "root",
		outcome: "not_found",
	});

	return (
		<RootDocument>
			<ErrorState
				title="page not found"
				detail="the route you requested does not exist."
			/>
		</RootDocument>
	);
}
