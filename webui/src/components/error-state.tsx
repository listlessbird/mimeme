interface ErrorStateProps {
	title: string;
	detail?: string;
	onRetry?: () => void;
}

export function ErrorState({ title, detail, onRetry }: ErrorStateProps) {
	return (
		<div className="min-h-screen bg-background p-4 md:p-6">
			<div className="mx-auto max-w-6xl">
				<div className="rounded-md border border-border bg-card p-6 text-center">
					<p className="text-sm text-foreground">{title}</p>
					{detail ? <p className="mt-2 text-xs text-muted-foreground">{detail}</p> : null}
					{onRetry ? (
						<button
							type="button"
							onClick={onRetry}
							className="mt-4 rounded-sm border border-border px-3 py-1.5 text-xs text-foreground transition-colors hover:bg-accent"
						>
							try again
						</button>
					) : null}
				</div>
			</div>
		</div>
	);
}
