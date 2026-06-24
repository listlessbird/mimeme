import { cn } from "@/lib/utils";
import { ImageOff } from "lucide-react";
import { useState } from "react";

interface ItemThumbnailProps {
	src: string | null;
	alt: string;
	className?: string;
}

export function ItemThumbnail({ src, alt, className }: ItemThumbnailProps) {
	const [failed, setFailed] = useState(false);

	if (!src || failed) {
		return (
			<div
				className={cn(
					"flex aspect-square items-center justify-center rounded-md bg-muted text-muted-foreground",
					className,
				)}
			>
				<ImageOff className="size-5" />
			</div>
		);
	}

	return (
		<img
			src={src}
			alt={alt}
			loading="lazy"
			onError={() => setFailed(true)}
			className={cn("aspect-square w-full rounded-md border bg-muted object-cover", className)}
		/>
	);
}
