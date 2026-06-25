import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface PaginationBarProps {
	offset: number;
	limit: number;
	total: number;
	loaded: number;
	isFetching?: boolean;
	onOffsetChange: (offset: number) => void;
}

export function PaginationBar({
	offset,
	limit,
	total,
	loaded,
	isFetching,
	onOffsetChange,
}: PaginationBarProps) {
	const from = total === 0 ? 0 : offset + 1;
	const to = offset + loaded;
	const hasPrev = offset > 0;
	const hasNext = offset + limit < total;

	return (
		<div className="flex items-center justify-between">
			<p className="text-xs text-muted-foreground tabular-nums">
				{from}–{to} of {total}
			</p>
			<div className="flex items-center gap-2">
				<Button
					variant="outline"
					size="sm"
					disabled={!hasPrev || isFetching}
					onClick={() => onOffsetChange(Math.max(0, offset - limit))}
				>
					<ChevronLeft data-icon="inline-start" />
					prev
				</Button>
				<Button
					variant="outline"
					size="sm"
					disabled={!hasNext || isFetching}
					onClick={() => onOffsetChange(offset + limit)}
				>
					next
					<ChevronRight data-icon="inline-end" />
				</Button>
			</div>
		</div>
	);
}
