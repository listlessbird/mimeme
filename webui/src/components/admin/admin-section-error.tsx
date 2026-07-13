import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { adminErrorMessage } from "@/lib/admin/api";
import { AlertTriangle, RotateCcw } from "lucide-react";

interface AdminSectionErrorProps {
	error: unknown;
	reset: () => void;
	title?: string;
}

export function AdminSectionError({
	error,
	reset,
	title = "section unavailable",
}: AdminSectionErrorProps) {
	return (
		<Alert variant="destructive">
			<AlertTriangle />
			<AlertTitle>{title}</AlertTitle>
			<AlertDescription className="flex flex-col items-start gap-3">
				<p>{adminErrorMessage(error, "the admin API could not be reached.")}</p>
				<Button type="button" size="sm" variant="outline" onClick={reset}>
					<RotateCcw data-icon="inline-start" />
					try again
				</Button>
			</AlertDescription>
		</Alert>
	);
}
