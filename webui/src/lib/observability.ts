export type LogLevel = "info" | "error";

interface LogEnvelope {
	level: LogLevel;
	event: string;
	timestamp: string;
	service: "webui";
	environment: {
		runtime: "server" | "client";
		region?: string;
		commit_hash?: string;
		service_version?: string;
	};
	data: Record<string, unknown>;
}

const isServer = typeof window === "undefined";

function environmentContext(): LogEnvelope["environment"] {
	if (!isServer) {
		return {
			runtime: "client",
			service_version: import.meta.env.VITE_APP_VERSION,
		};
	}

	return {
		runtime: "server",
		region: process.env.CF_REGION ?? process.env.AWS_REGION,
		commit_hash:
			process.env.COMMIT_SHA ??
			process.env.CF_PAGES_COMMIT_SHA ??
			process.env.VERCEL_GIT_COMMIT_SHA,
		service_version: process.env.npm_package_version,
	};
}

function emit(
	level: LogLevel,
	event: string,
	data: Record<string, unknown>,
): void {
	const payload: LogEnvelope = {
		level,
		event,
		timestamp: new Date().toISOString(),
		service: "webui",
		environment: environmentContext(),
		data,
	};

	if (level === "error") {
		console.error(JSON.stringify(payload));
		return;
	}

	console.info(JSON.stringify(payload));
}

export function logInfo(event: string, data: Record<string, unknown>): void {
	emit("info", event, data);
}

export function logError(event: string, data: Record<string, unknown>): void {
	emit("error", event, data);
}

export function serializeError(error: unknown): Record<string, unknown> {
	if (error instanceof Error) {
		return {
			type: error.name,
			message: error.message,
			stack: error.stack,
		};
	}

	return {
		type: "UnknownError",
		message: String(error),
	};
}
