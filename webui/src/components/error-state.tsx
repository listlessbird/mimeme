import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Spinner } from "@/components/ui/spinner";
import { RotateCw } from "lucide-react";

interface ErrorStateProps {
	title: string;
	detail?: string;
	onRetry?: () => void;
	isRetrying?: boolean;
}

export function ErrorState({ title, detail, onRetry, isRetrying = false }: ErrorStateProps) {
	return (
		<main className="flex min-h-screen items-start justify-center bg-background p-4 pt-[clamp(5rem,18vh,10rem)] md:p-6 md:pt-[clamp(7rem,22vh,12rem)]">
			<div className="w-full max-w-xl">
				<Card
					role="alert"
					aria-busy={isRetrying}
					className="gap-0 overflow-hidden rounded-md border-destructive/40 py-0 text-center shadow-none"
				>
					<CardHeader className="gap-2 px-4 py-6 sm:px-6">
						<CardTitle className="text-base text-destructive">{title}</CardTitle>
						{detail ? <CardDescription className="text-xs">{detail}</CardDescription> : null}
					</CardHeader>
					{onRetry ? (
						<CardFooter className="justify-center border-t px-4 py-4 sm:px-6">
							<Button
								type="button"
								variant="outline"
								disabled={isRetrying}
								onClick={onRetry}
								className="transition-[background-color,box-shadow,transform] active:scale-[0.96]"
							>
								{isRetrying ? (
									<Spinner data-icon="inline-start" />
								) : (
									<RotateCw data-icon="inline-start" />
								)}
								{isRetrying ? "retrying…" : "retry search"}
							</Button>
						</CardFooter>
					) : null}
				</Card>
			</div>
		</main>
	);
}
