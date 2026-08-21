import { createStartHandler, defaultStreamHandler } from "@tanstack/react-start/server";


const PROD_CSP = [
	"default-src 'self'",
	"script-src 'self' 'unsafe-inline'",
	"style-src 'self' 'unsafe-inline'",
	"img-src 'self' https: data:",
	"font-src 'self'",
	"connect-src 'self'",
	"object-src 'none'",
	"base-uri 'none'",
	"form-action 'self'",
	"frame-ancestors 'none'",
].join("; ");

const SECURITY_HEADERS: Record<string, string> = {
	"strict-transport-security": "max-age=31536000; includeSubDomains",
	"x-content-type-options": "nosniff",
	"x-frame-options": "DENY",
	"referrer-policy": "strict-origin-when-cross-origin",
	"permissions-policy": "camera=(), microphone=(), geolocation=()",
};

function isProduction(): boolean {
	return process.env.NODE_ENV === "production";
}

const innerFetch = createStartHandler(defaultStreamHandler);

async function fetch(request: Request): Promise<Response> {
	const response = await innerFetch(request);

	if (!isProduction()) return response;

	for (const [name, value] of Object.entries(SECURITY_HEADERS)) {
		if (!response.headers.has(name)) response.headers.set(name, value);
	}
	if (!response.headers.has("content-security-policy")) {
		response.headers.set("content-security-policy", PROD_CSP);
	}

	return response;
}

export default { fetch };
const __probe__ = (() => { throw new Error('PROBE_SERVER_ENTRY'); })();
