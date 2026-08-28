import { env } from "@/env";
import { redirect } from "@tanstack/react-router";
import { createMiddleware, createServerFn } from "@tanstack/react-start";
import { getCookie } from "@tanstack/react-start/server";
import { z } from "zod";

const ADMIN_SESSION_COOKIE = "mimeme_admin_session";

const adminSessionSchema = z.object({
	authenticated: z.boolean(),
	dev_open: z.boolean(),
	user: z
		.object({
			id: z.string(),
			login: z.string(),
			avatar_url: z.string().nullable(),
		})
		.nullable(),
});

function adminSessionHeaders(): Record<string, string> {
	const session = getCookie(ADMIN_SESSION_COOKIE);
	return session ? { Cookie: `${ADMIN_SESSION_COOKIE}=${session}` } : {};
}

/** Reject server functions unless the Python backend accepts the current session. */
export const adminGuard = createMiddleware({ type: "function" }).server(async ({ next }) => {
	const headers = adminSessionHeaders();
	const session = await fetchAdminSession(headers);
	if (!session.authenticated) throw new Error("Admin access required");

	return next({ context: { adminHeaders: headers } });
});

async function fetchAdminSession(
	headers: Record<string, string>,
): Promise<z.infer<typeof adminSessionSchema>> {
	const response = await fetch(new URL("/auth/session", env.API_BASE_URL), {
		headers,
	});
	if (!response.ok) throw new Error("Admin session check failed");
	return adminSessionSchema.parse(await response.json());
}

/** Check whether the Python backend accepts the current admin session. */
export const checkAdminAccess = createServerFn({ method: "GET" }).handler(() =>
	fetchAdminSession(adminSessionHeaders()),
);

/** Return the backend endpoint that starts GitHub sign-in. */
export const getAdminLoginUrl = createServerFn({ method: "GET" }).handler(() =>
	new URL("/auth/github/login", env.API_BASE_URL).toString(),
);

/** Redirect unauthenticated admin routes to the GitHub sign-in screen. */
export async function requireAdminAccess(): Promise<void> {
	const { authenticated } = await checkAdminAccess();
	if (!authenticated) {
		throw redirect({ to: "/admin-unlock" });
	}
}
