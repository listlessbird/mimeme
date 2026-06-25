import type { SearchResult } from "@/lib/api";
import { motion, useReducedMotion } from "motion/react";
import { memo, useCallback, useState } from "react";

interface MemeCardProps {
	meme: SearchResult;
	onSelect: (meme: SearchResult) => void;
}

const easeOutQuint = [0.23, 1, 0.32, 1] as const;
const cardTransition = { duration: 0.2, ease: easeOutQuint };
const hoverScale = { scale: 1.02 };
const tapScale = { scale: 0.97 };
const cardVariants = {
	hidden: { opacity: 0, y: 12 },
	show: { opacity: 1, y: 0 },
};
const cardVariantsReduced = {
	hidden: { opacity: 1, y: 0 },
	show: { opacity: 1, y: 0 },
};
const reducedImgStyle = { opacity: 1, transform: "none", transition: "none" } as const;

export const MemeCard = memo(function MemeCard({ meme, onSelect }: MemeCardProps) {
	const [loaded, setLoaded] = useState(false);
	const shouldReduceMotion = useReducedMotion();

	const handleClick = useCallback(() => onSelect(meme), [onSelect, meme]);
	const handleLoad = useCallback(() => setLoaded(true), []);

	return (
		<motion.button
			onClick={handleClick}
			className="group relative w-full overflow-hidden rounded-md border border-border bg-card text-left transition-colors will-change-transform hover:border-foreground/30"
			variants={shouldReduceMotion ? cardVariantsReduced : cardVariants}
			transition={cardTransition}
			whileHover={shouldReduceMotion ? undefined : hoverScale}
			whileTap={shouldReduceMotion ? undefined : tapScale}
		>
			<div className="relative aspect-square overflow-hidden">
				{!loaded && <div className="shimmer absolute inset-0 rounded-sm" />}
				<img
					src={meme.url}
					alt={meme.caption || "meme"}
					loading="lazy"
					onLoad={handleLoad}
					className={`h-full w-full object-cover transition-all duration-500 ease-out ${
						loaded ? "scale-100 opacity-100" : "scale-[1.03] opacity-0"
					}`}
					style={shouldReduceMotion ? reducedImgStyle : undefined}
				/>
			</div>

			<div className="absolute inset-0 flex items-end bg-background/80 p-3 opacity-0 transition-opacity group-hover:opacity-100">
				<span className="line-clamp-2 text-xs text-foreground">{meme.caption}</span>
			</div>
		</motion.button>
	);
});
