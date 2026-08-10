import { MemeCard } from "@/components/meme-card";
import { MemeModal } from "@/components/meme-modal";
import type { SearchResponse, SearchResult } from "@/lib/api";
import { motion, AnimatePresence, useReducedMotion } from "motion/react";
import { useCallback, useState } from "react";

interface MemeGridProps {
	data: SearchResponse;
}

const easeOutQuint = [0.23, 1, 0.32, 1] as const;
const emptyInitial = { opacity: 0, y: 8 };
const emptyAnimate = { opacity: 1, y: 0 };
const emptyTransition = { duration: 0.25, ease: easeOutQuint };
const gridVariants = {
	hidden: {},
	show: { transition: { staggerChildren: 0.04 } },
};
const gridVariantsReduced = {
	hidden: {},
	show: { transition: { staggerChildren: 0 } },
};

export function MemeGrid({ data }: MemeGridProps) {
	const [selectedMeme, setSelectedMeme] = useState<SearchResult | null>(null);
	const shouldReduceMotion = useReducedMotion();

	const handleSelect = useCallback((meme: SearchResult) => setSelectedMeme(meme), []);
	const handleClose = useCallback(() => setSelectedMeme(null), []);

	if (data.results.length === 0) {
		return (
			<motion.div
				className="py-20 text-center text-sm text-muted-foreground"
				initial={shouldReduceMotion ? false : emptyInitial}
				animate={emptyAnimate}
				transition={emptyTransition}
			>
				<p>no results for "{data.query}"</p>
				<p className="mt-2 text-xs">try shorter phrases or different keywords</p>
			</motion.div>
		);
	}

	return (
		<>
			<motion.div
				className="grid min-w-0 grid-cols-2 gap-3 sm:grid-cols-3 sm:gap-4 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6"
				key={data.query}
				initial="hidden"
				animate="show"
				variants={shouldReduceMotion ? gridVariantsReduced : gridVariants}
			>
				{data.results.map((meme) => (
					<MemeCard key={meme.id} meme={meme} onSelect={handleSelect} />
				))}
			</motion.div>

			<AnimatePresence>
				{selectedMeme && <MemeModal meme={selectedMeme} onClose={handleClose} />}
			</AnimatePresence>
		</>
	);
}
