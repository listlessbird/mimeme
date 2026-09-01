import { createMiddleware, createStart } from "@tanstack/react-start";

function isProduction(): boolean {
	return process.env.NODE_ENV === "production";
}

const serverFnOriginGuard = createMiddleware().server(async ({ request, next }) => {
	const { origin, pathname } = new URL(request.url);
	const isServerFnCall = pathname.startsWith("/_serverFn/");
	const stateChanging = !["GET", "HEAD", "OPTIONS"].includes(request.method);

	if (isProduction() && isServerFnCall && stateChanging && origin) {
		if (origin !== new URL(request.url).origin) {
			return new Response("Cross-origin server function call rejected", { status: 403 });
		}
	}

	return next();
});

export const startInstance = createStart(() => ({
	requestMiddleware: [serverFnOriginGuard],
}));
