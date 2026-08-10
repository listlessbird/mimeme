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
const tapScale = { scale: 0.96 };
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
			type="button"
			onClick={handleClick}
			aria-label={`Open meme${meme.caption ? `: ${meme.caption}` : ""}`}
			className="group relative w-full min-w-0 overflow-hidden rounded-lg border border-border bg-card text-left transition-[border-color,box-shadow] hover:border-foreground/30 hover:shadow-lg focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
			variants={shouldReduceMotion ? cardVariantsReduced : cardVariants}
			transition={cardTransition}
			whileHover={shouldReduceMotion ? undefined : hoverScale}
			whileTap={shouldReduceMotion ? undefined : tapScale}
		>
			<div
				className="relative w-full overflow-hidden bg-muted/20"
				style={{ aspectRatio: `${meme.width} / ${meme.height}` }}
			>
				{!loaded && <div className="shimmer absolute inset-0 rounded-sm" />}
				<img
					src={meme.url}
					alt={meme.caption || "meme"}
					loading="lazy"
					onLoad={handleLoad}
					className={`absolute inset-0 h-full w-full object-contain outline-1 outline-white/10 transition-[opacity,transform] duration-500 ease-out ${
						loaded ? "scale-100 opacity-100" : "scale-[1.03] opacity-0"
					}`}
					style={shouldReduceMotion ? reducedImgStyle : undefined}
				/>
			</div>

			{meme.caption ? (
				<div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 via-black/35 to-transparent px-3 pt-10 pb-3 opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100">
					<span className="line-clamp-2 text-xs leading-relaxed text-white">{meme.caption}</span>
				</div>
			) : null}
		</motion.button>
	);
});
