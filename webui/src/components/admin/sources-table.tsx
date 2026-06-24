import {
	Table,
	TableBody,
	TableCell,
	TableHead,
	TableHeader,
	TableRow,
} from "@/components/ui/table";
import type { SourceListItem } from "@/lib/admin/api";
import { scheduleLabel } from "@/lib/admin/format";
import { Link } from "@tanstack/react-router";

import { EnabledBadge } from "./badges";

export function SourcesTable({ sources }: { sources: SourceListItem[] }) {
	return (
		<div className="rounded-md border">
			<Table>
				<TableHeader>
					<TableRow>
						<TableHead>name</TableHead>
						<TableHead>adapter</TableHead>
						<TableHead>state</TableHead>
						<TableHead>schedule</TableHead>
						<TableHead className="text-right">runs</TableHead>
						<TableHead className="text-right">items</TableHead>
						<TableHead className="text-right">duplicates</TableHead>
					</TableRow>
				</TableHeader>
				<TableBody>
					{sources.map((source) => (
						<TableRow key={source.id}>
							<TableCell className="font-medium">
								<Link
									to="/admin/sources/$id"
									params={{ id: String(source.id) }}
									className="hover:underline"
								>
									{source.name}
								</Link>
							</TableCell>
							<TableCell className="text-muted-foreground">{source.adapter_key}</TableCell>
							<TableCell>
								<EnabledBadge enabled={source.enabled} />
							</TableCell>
							<TableCell className="text-muted-foreground">
								{scheduleLabel(source.schedule_cron, source.schedule_timezone)}
							</TableCell>
							<TableCell className="text-right tabular-nums">{source.stats.run_count}</TableCell>
							<TableCell className="text-right tabular-nums">
								{source.stats.items_discovered}
							</TableCell>
							<TableCell className="text-right tabular-nums">
								{source.stats.duplicate_count}
							</TableCell>
						</TableRow>
					))}
				</TableBody>
			</Table>
		</div>
	);
}
