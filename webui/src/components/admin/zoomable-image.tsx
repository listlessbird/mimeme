import { Dialog, DialogClose, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import { ImageOff, X } from "lucide-react";
import { useState } from "react";

interface ZoomableImageProps {
	src: string | null;
	alt: string;
	className?: string;
	triggerClassName?: string;
	fallbackClassName?: string;
}

export function ZoomableImage({
	src,
	alt,
	className,
	triggerClassName,
	fallbackClassName,
}: ZoomableImageProps) {
	const [failed, setFailed] = useState(false);
	const [open, setOpen] = useState(false);

	if (!src || failed) {
		return (
			<div
				className={cn(
					"flex items-center justify-center rounded-md border bg-muted text-muted-foreground",
					fallbackClassName ?? className,
				)}
			>
				<ImageOff className="size-5" />
			</div>
		);
	}

	return (
		<>
			<button
				type="button"
				onClick={() => setOpen(true)}
				aria-label={`expand ${alt}`}
				className={cn(
					"group inline-flex rounded-md focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none",
					triggerClassName,
				)}
			>
				<img
					src={src}
					alt={alt}
					loading="lazy"
					onError={() => setFailed(true)}
					className={cn("transition-opacity group-hover:opacity-90", className)}
				/>
			</button>

			<Dialog open={open} onOpenChange={setOpen}>
				<DialogContent
					showCloseButton={false}
					aria-describedby={undefined}
					className="flex w-fit max-w-[95vw] items-center justify-center border-0 bg-transparent p-0 shadow-none sm:max-w-[95vw]"
				>
					<DialogTitle className="sr-only">{alt}</DialogTitle>
					<img
						src={src}
						alt={alt}
						className="max-h-[90vh] w-auto max-w-[95vw] rounded-md object-contain"
					/>
					<DialogClose className="absolute top-2 right-2 rounded-full bg-black/60 p-1.5 text-white transition-colors hover:bg-black/80 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none">
						<X className="size-4" />
						<span className="sr-only">close</span>
					</DialogClose>
				</DialogContent>
			</Dialog>
		</>
	);
}
