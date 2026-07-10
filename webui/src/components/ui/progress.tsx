"use client";

import { cn } from "@/lib/utils";
import { Progress as ProgressPrimitive } from "@base-ui/react/progress";
import * as React from "react";

function Progress({
	className,
	value,
	...props
}: React.ComponentProps<typeof ProgressPrimitive.Root>) {
	return (
		<ProgressPrimitive.Root
			data-slot="progress"
			className={cn("w-full", className)}
			value={value}
			{...props}
		>
			<ProgressPrimitive.Track
				data-slot="progress-track"
				className="relative h-2 w-full overflow-hidden rounded-full bg-primary/20"
			>
				<ProgressPrimitive.Indicator
					data-slot="progress-indicator"
					className="h-full bg-primary transition-[width]"
				/>
			</ProgressPrimitive.Track>
		</ProgressPrimitive.Root>
	);
}

export { Progress };
