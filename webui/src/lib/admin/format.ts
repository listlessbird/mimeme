import { format, formatDistanceToNow } from "date-fns";

export function relativeTime(iso: string | null): string {
	if (!iso) return "—";
	try {
		return formatDistanceToNow(new Date(iso), { addSuffix: true });
	} catch {
		return "—";
	}
}

export function absoluteTime(iso: string | null): string {
	if (!iso) return "—";
	try {
		return format(new Date(iso), "yyyy-MM-dd HH:mm");
	} catch {
		return "—";
	}
}

export function scheduleLabel(cron: string | null, timezone: string): string {
	if (!cron) return "manual only";
	return `${cron} · ${timezone}`;
}
