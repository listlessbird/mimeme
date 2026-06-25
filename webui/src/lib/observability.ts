export type LogLevel = "info" | "error";

import { initLogger, log, parseError } from "evlog";

const isServer = typeof window === "undefined";
const runtimeNodeEnv =
	isServer && typeof process !== "undefined" ? process.env.NODE_ENV : import.meta.env.MODE;
let loggerInitialized = false;

function isPrettyLoggingEnabled(): boolean {
	if (isServer) {
		return process.env.LOG_PRETTY === "1" || runtimeNodeEnv !== "production";
	}

	return import.meta.env.VITE_LOG_PRETTY === "1" || runtimeNodeEnv !== "production";
}

function environmentContext(): {
	region?: string;
	commitHash?: string;
	version?: string;
} {
	if (!isServer) return {};

	return {
		region: process.env.CF_REGION ?? process.env.AWS_REGION,
		commitHash:
			process.env.COMMIT_SHA ??
			process.env.CF_PAGES_COMMIT_SHA ??
			process.env.VERCEL_GIT_COMMIT_SHA,
		version: process.env.npm_package_version,
	};
}

function initializeLogger(): void {
	if (loggerInitialized) return;

	initLogger({
		env: {
			service: "webui",
			environment: runtimeNodeEnv ?? "development",
			...environmentContext(),
		},
		pretty: isPrettyLoggingEnabled(),
	});

	loggerInitialized = true;
}

function emit(level: LogLevel, event: string, data: Record<string, unknown>): void {
	initializeLogger();

	const payload = {
		event,
		...data,
	};

	if (level === "error") {
		log.error(payload);
		return;
	}

	log.info(payload);
}

export function logInfo(event: string, data: Record<string, unknown>): void {
	emit("info", event, data);
}

export function logError(event: string, data: Record<string, unknown>): void {
	emit("error", event, data);
}

export function serializeError(error: unknown): Record<string, unknown> {
	const parsedError = parseError(error);

	if (error instanceof Error) {
		return {
			type: error.name,
			message: error.message,
			stack: error.stack,
			status: parsedError.status,
			why: parsedError.why,
			fix: parsedError.fix,
			link: parsedError.link,
		};
	}

	return {
		type: "UnknownError" as const,
		message: parsedError.message,
		status: parsedError.status,
		why: parsedError.why,
		fix: parsedError.fix,
		link: parsedError.link,
	};
}
