import { ErrorState } from "@/components/error-state";
import { logError, logInfo, serializeError } from "@/lib/observability";
import { TanStackDevtools } from "@tanstack/react-devtools";
import type { QueryClient } from "@tanstack/react-query";
import { createRootRouteWithContext, HeadContent, Scripts } from "@tanstack/react-router";
import { TanStackRouterDevtoolsPanel } from "@tanstack/react-router-devtools";
import { Agentation } from "agentation";
import { NuqsAdapter } from "nuqs/adapters/tanstack-router";

import TanStackQueryDevtools from "../integrations/tanstack-query/devtools";

import appCss from "@/styles.css?url";

interface MyRouterContext {
	queryClient: QueryClient;
}

const siteTitle = "mìmeme";
const siteDescription = "Find that meme.";
const siteUrl = import.meta.env.VITE_SITE_URL?.replace(/\/$/, "") ?? "";
const canonicalUrl = siteUrl || "/";
const ogImageUrl = siteUrl ? `${siteUrl}/og-image.jpg` : "/og-image.jpg";
const ogImageAlt = "mimeme jester logo with find that meme text";

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
				title: siteTitle,
			},
			{
				name: "description",
				content: siteDescription,
			},
			{
				property: "og:title",
				content: siteTitle,
			},
			{
				property: "og:description",
				content: siteDescription,
			},
			{
				property: "og:type",
				content: "website",
			},
			{
				property: "og:url",
				content: canonicalUrl,
			},
			{
				property: "og:site_name",
				content: siteTitle,
			},
			{
				property: "og:image",
				content: ogImageUrl,
			},
			{
				property: "og:image:type",
				content: "image/jpeg",
			},
			{
				property: "og:image:width",
				content: "1200",
			},
			{
				property: "og:image:height",
				content: "630",
			},
			{
				property: "og:image:alt",
				content: ogImageAlt,
			},
			{
				name: "twitter:card",
				content: "summary_large_image",
			},
			{
				name: "twitter:title",
				content: siteTitle,
			},
			{
				name: "twitter:description",
				content: siteDescription,
			},
			{
				name: "twitter:image",
				content: ogImageUrl,
			},
			{
				name: "twitter:image:alt",
				content: ogImageAlt,
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
	return (
		<html lang="en">
			<head>
				<HeadContent />
			</head>
			<body className="font-mono antialiased">
				<NuqsAdapter>{children}</NuqsAdapter>
				<TanStackDevtools
					config={{
						position: "bottom-left",
					}}
					plugins={[
						{
							name: "Tanstack Router",
							render: <TanStackRouterDevtoolsPanel />,
						},
						TanStackQueryDevtools,
					]}
				/>
				{import.meta.env.DEV && <Agentation />}
				<Scripts />
			</body>
		</html>
	);
}

function RootErrorComponent({ error, reset }: { error: unknown; reset: () => void }) {
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
			<ErrorState title="page not found" detail="the route you requested does not exist." />
		</RootDocument>
	);
}
