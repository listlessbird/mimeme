import { env } from "@/env";
import { isAccessAllowed } from "@/lib/admin/guard-logic";
import { redirect } from "@tanstack/react-router";
import { createMiddleware, createServerFn } from "@tanstack/react-start";
import { getCookie, setCookie } from "@tanstack/react-start/server";

export const ADMIN_SESSION_COOKIE = "admin_ui_session";
const SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 7;

function isProduction(): boolean {
	return process.env.NODE_ENV === "production";
}

function hasValidSession(): boolean {
	return isAccessAllowed({
		isProduction: isProduction(),
		secret: env.ADMIN_UI_SECRET,
		cookie: getCookie(ADMIN_SESSION_COOKIE),
	});
}


export const adminGuard = createMiddleware({ type: "function" }).server(async ({ next }) => {
	if (!isProduction()) return next();

	if (!env.ADMIN_UI_SECRET) throw new Error("Admin access is not configured");

	if (!hasValidSession()) throw new Error("Admin access required");

	return next();
});

export const checkAdminAccess = createServerFn({ method: "GET" }).handler(async () => ({
	allowed: hasValidSession(),
	devOpen: !isProduction(),
}));

export const unlockAdmin = createServerFn({ method: "POST" })
	.inputValidator((input: { secret: string }) => input)
	.handler(async ({ data }) => {
		if (!isProduction()) return { ok: true as const };

		if (!env.ADMIN_UI_SECRET || data.secret !== env.ADMIN_UI_SECRET) {
			return { ok: false as const };
		}

		setCookie(ADMIN_SESSION_COOKIE, env.ADMIN_UI_SECRET, {
			httpOnly: true,
			secure: true,
			sameSite: "lax",
			path: "/",
			maxAge: SESSION_MAX_AGE_SECONDS,
		});
		return { ok: true as const };
	});

export async function requireAdminAccess(): Promise<void> {
	const { allowed } = await checkAdminAccess();
	if (!allowed) {
		throw redirect({ to: "/admin-unlock" });
	}
}
