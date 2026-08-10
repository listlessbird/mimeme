import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import type { SearchResult } from "@/lib/api";
import { cn } from "@/lib/utils";
import { memo, useCallback, useState } from "react";

interface MemeCardProps {
	meme: SearchResult;
	onSelect: (meme: SearchResult) => void;
}

export const MemeCard = memo(function MemeCard({ meme, onSelect }: MemeCardProps) {
	const [loaded, setLoaded] = useState(false);

	const handleClick = useCallback(() => onSelect(meme), [onSelect, meme]);
	const handleLoad = useCallback(() => setLoaded(true), []);

	return (
		<Button
			type="button"
			onClick={handleClick}
			aria-label={`Open meme${meme.caption ? `: ${meme.caption}` : ""}`}
			variant="ghost"
			className="group relative h-auto w-full min-w-0 overflow-hidden rounded-lg border border-border bg-card p-0 text-left shadow-none transition-[border-color,box-shadow,transform] hover:bg-card focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none active:scale-[0.96] [@media(hover:hover)_and_(pointer:fine)]:hover:border-foreground/30 [@media(hover:hover)_and_(pointer:fine)]:hover:shadow-lg"
		>
			<div className="relative aspect-[4/5] w-full overflow-hidden bg-muted">
				{!loaded ? <Skeleton className="absolute inset-0 rounded-none" aria-hidden="true" /> : null}
				<img
					src={meme.url}
					alt={meme.caption || "meme"}
					width={meme.width}
					height={meme.height}
					loading="lazy"
					onLoad={handleLoad}
					className={cn(
						"absolute inset-0 size-full object-cover transition-[opacity,transform] duration-200 ease-out motion-reduce:transition-none",
						loaded ? "scale-100 opacity-100" : "scale-[1.01] opacity-0",
					)}
				/>
			</div>

			{meme.caption ? (
				<div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 via-black/35 to-transparent px-3 pt-10 pb-3 opacity-0 transition-opacity duration-150 group-focus-visible:opacity-100 [@media(hover:hover)_and_(pointer:fine)]:group-hover:opacity-100">
					<span className="line-clamp-2 text-xs leading-relaxed text-white">{meme.caption}</span>
				</div>
			) : null}
		</Button>
	);
});
