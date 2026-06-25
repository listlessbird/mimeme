import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { Link, type LinkComponentProps } from "@tanstack/react-router";
import { ArrowUpRight } from "lucide-react";

import { CopyButton } from "./copy-button";

export type ChipLink = Pick<LinkComponentProps, "to" | "params">;

interface IdChipProps {
	label: string;
	value: string | number | null | undefined;
	link?: ChipLink;
	truncate?: boolean;
	className?: string;
}

export function IdChip({ label, value, link, truncate, className }: IdChipProps) {
	if (value == null || value === "") return null;
	const text = String(value);

	const valueNode = link ? (
		<Link
			{...link}
			className="inline-flex items-center gap-0.5 font-mono hover:underline"
			onClick={(e) => e.stopPropagation()}
		>
			<span className={cn(truncate && "max-w-[12ch] truncate")}>{text}</span>
			<ArrowUpRight className="size-3 shrink-0" />
		</Link>
	) : (
		<span className={cn("font-mono", truncate && "max-w-[16ch] truncate")}>{text}</span>
	);

	return (
		<Badge
			variant="outline"
			className={cn("gap-1 py-0 pr-0.5 pl-2 font-normal", className)}
			title={`${label}: ${text}`}
		>
			<span className="text-muted-foreground">{label}</span>
			{valueNode}
			<CopyButton value={text} label={label} className="size-5 text-muted-foreground" />
		</Badge>
	);
}
