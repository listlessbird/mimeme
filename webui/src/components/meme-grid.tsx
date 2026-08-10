import { MemeCard } from "@/components/meme-card";
import { MemeModal } from "@/components/meme-modal";
import { Skeleton } from "@/components/ui/skeleton";
import type { SearchResponse, SearchResult } from "@/lib/api";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useCallback, useState } from "react";

interface MemeGridProps {
	data: SearchResponse;
}

export function MemeGrid({ data }: MemeGridProps) {
	const [selectedMeme, setSelectedMeme] = useState<SearchResult | null>(null);
	const shouldReduceMotion = useReducedMotion();

	const handleSelect = useCallback((meme: SearchResult) => setSelectedMeme(meme), []);
	const handleClose = useCallback(() => setSelectedMeme(null), []);

	if (data.results.length === 0) {
		return (
			<motion.div
				className="py-20 text-center text-sm text-muted-foreground"
				initial={shouldReduceMotion ? false : { opacity: 0, y: 8 }}
				animate={{ opacity: 1, y: 0 }}
				transition={{ duration: 0.2, ease: [0.23, 1, 0.32, 1] }}
			>
				<p>no results for "{data.query}"</p>
				<p className="mt-2 text-xs">try shorter phrases or different keywords</p>
			</motion.div>
		);
	}

	return (
		<>
			<div className="@container/results">
				<div className="grid min-w-0 grid-cols-[repeat(auto-fit,minmax(min(100%,15rem),1fr))] gap-3 @[40rem]/results:gap-4">
					{data.results.map((meme) => (
						<MemeCard key={meme.id} meme={meme} onSelect={handleSelect} />
					))}
				</div>
			</div>

			<AnimatePresence>
				{selectedMeme && <MemeModal meme={selectedMeme} onClose={handleClose} />}
			</AnimatePresence>
		</>
	);
}

export function MemeGridSkeleton() {
	return (
		<div className="@container/results">
			<div className="grid min-w-0 grid-cols-[repeat(auto-fit,minmax(min(100%,15rem),1fr))] gap-3 @[40rem]/results:gap-4">
				{Array.from({ length: 10 }).map((_, i) => (
					<Skeleton key={i} className="aspect-[4/5] w-full rounded-lg" />
				))}
			</div>
		</div>
	);
}
