import type { SearchResult } from "@/lib/api";
import { X, Download, Copy, Check } from "lucide-react";
import { motion, useReducedMotion } from "motion/react";
import { useEffect, useCallback, useState } from "react";

interface MemeModalProps {
	meme: SearchResult;
	onClose: () => void;
}

const easeOutQuint = [0.23, 1, 0.32, 1] as const;
const baseTransition = { duration: 0.2, ease: easeOutQuint };
const fadeFrom = { opacity: 0 };
const fadeTo = { opacity: 1 };
const fadeExit = { opacity: 0, transition: { duration: 0.15, ease: "easeIn" } } as const;
const panelInitial = { opacity: 0, scale: 0.95, y: 20 };
const panelAnimate = { opacity: 1, scale: 1, y: 0 };
const panelExit = {
	opacity: 0,
	scale: 0.97,
	y: 10,
	transition: { duration: 0.15, ease: "easeIn" },
} as const;

export function MemeModal({ meme, onClose }: MemeModalProps) {
	const shouldReduceMotion = useReducedMotion();
	const [copied, setCopied] = useState(false);

	const handleKeyDown = useCallback(
		(e: KeyboardEvent) => {
			if (e.key === "Escape") onClose();
		},
		[onClose],
	);

	useEffect(() => {
		document.addEventListener("keydown", handleKeyDown);
		document.body.style.overflow = "hidden";
		return () => {
			document.removeEventListener("keydown", handleKeyDown);
			document.body.style.overflow = "unset";
		};
	}, [handleKeyDown]);

	const handleCopy = useCallback(async () => {
		await navigator.clipboard.writeText(meme.url);
		setCopied(true);
		setTimeout(() => setCopied(false), 1500);
	}, [meme.url]);

	const handleSave = useCallback(() => {
		window.open(meme.url, "_blank");
	}, [meme.url]);

	return (
		<motion.div
			className="fixed inset-0 z-50 flex items-end justify-center sm:items-center"
			initial={shouldReduceMotion ? false : fadeFrom}
			animate={fadeTo}
			exit={fadeExit}
			transition={baseTransition}
		>
			<motion.div
				className="absolute inset-0 bg-background/90 backdrop-blur-sm"
				onClick={onClose}
				initial={shouldReduceMotion ? false : fadeFrom}
				animate={fadeTo}
				exit={fadeExit}
				transition={baseTransition}
			/>

			<motion.div
				className="relative z-10 w-full max-w-2xl overflow-hidden rounded-t-lg border border-border bg-card will-change-transform sm:rounded-lg"
				initial={shouldReduceMotion ? false : panelInitial}
				animate={panelAnimate}
				exit={panelExit}
				transition={baseTransition}
			>
				{/* Header — close only */}
				<div className="flex items-center justify-end border-b border-border p-4">
					<button
						onClick={onClose}
						className="-mr-2 p-2 text-muted-foreground transition-colors hover:text-foreground"
					>
						<X className="h-4 w-4" />
					</button>
				</div>

				{/* Image */}
				<div className="p-4">
					<img
						src={meme.url}
						alt={meme.caption || "meme"}
						className="max-h-[60vh] w-full rounded-sm object-contain"
					/>
				</div>

				{/* Info — bottom panel */}
				<div className="space-y-2 px-4 pt-3 pb-4">
					{meme.caption && (
						<p className="text-sm leading-relaxed text-foreground">{meme.caption}</p>
					)}

					{meme.ocr_text && (
						<p className="text-xs leading-relaxed text-muted-foreground italic">
							"{meme.ocr_text}"
						</p>
					)}

					<div className="flex items-center justify-between border-t border-border pt-2">
						<span className="text-[10px] tracking-wider text-muted-foreground/60 uppercase tabular-nums">
							{meme.score.toFixed(3)} · {meme.width}×{meme.height}
						</span>

						<div className="flex items-center gap-1">
							<ActionButton label={copied ? "copied" : "copy"} onClick={handleCopy}>
								{copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
							</ActionButton>
							<ActionButton label="save" onClick={handleSave}>
								<Download className="h-3 w-3" />
							</ActionButton>
						</div>
					</div>
				</div>
			</motion.div>
		</motion.div>
	);
}

function ActionButton({
	children,
	label,
	onClick,
}: {
	children: React.ReactNode;
	label: string;
	onClick: () => void;
}) {
	return (
		<button
			onClick={onClick}
			className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-secondary/80 hover:text-foreground"
		>
			{children}
			{label}
		</button>
	);
}
