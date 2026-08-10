import { Button } from "@/components/ui/button";
import { Dialog, DialogClose, DialogContent, DialogTitle } from "@/components/ui/dialog";
import type { SearchResult } from "@/lib/api";
import { Check, Copy, ExternalLink, X } from "lucide-react";
import { useCallback, useState } from "react";

interface MemeModalProps {
	meme: SearchResult;
	onClose: () => void;
}

export function MemeModal({ meme, onClose }: MemeModalProps) {
	const [copied, setCopied] = useState(false);

	const handleCopy = useCallback(async () => {
		try {
			await navigator.clipboard.writeText(meme.url);
			setCopied(true);
			window.setTimeout(() => setCopied(false), 1500);
		} catch {
			setCopied(false);
		}
	}, [meme.url]);

	const handleSave = useCallback(() => {
		window.open(meme.url, "_blank", "noopener,noreferrer");
	}, [meme.url]);

	return (
		<Dialog open onOpenChange={(open) => !open && onClose()}>
			<DialogContent
				showCloseButton={false}
				aria-describedby={undefined}
				className="flex max-h-[calc(100dvh-2rem)] w-fit max-w-[calc(100%-1rem)] flex-col items-center justify-center gap-3 border-0 bg-transparent p-2 shadow-none sm:max-w-[calc(100%-3rem)] sm:p-4"
			>
				<DialogTitle className="sr-only">Meme preview</DialogTitle>
				<img
					src={meme.url}
					alt={meme.caption || "Meme preview"}
					className="max-h-[calc(100dvh-9rem)] w-auto max-w-full rounded-lg object-contain outline-1 outline-white/10"
				/>

				<div className="flex items-center gap-1 rounded-full border border-white/10 bg-black/70 p-1 text-white shadow-xl backdrop-blur-md">
					<ActionButton label={copied ? "Copied image URL" : "Copy image URL"} onClick={handleCopy}>
						{copied ? <Check className="size-4" /> : <Copy className="size-4" />}
					</ActionButton>
					<ActionButton label="Open image in a new tab" onClick={handleSave}>
						<ExternalLink className="size-4" />
					</ActionButton>
				</div>

				<DialogClose
					className="absolute top-2 right-2 rounded-full bg-black/70 p-2 text-white shadow-lg transition-[background-color,transform] hover:bg-black/90 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none active:scale-[0.96] sm:top-4 sm:right-4"
					aria-label="Close image viewer"
				>
					<X className="size-5" />
					<span className="sr-only">Close image viewer</span>
				</DialogClose>
			</DialogContent>
		</Dialog>
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
		<Button
			type="button"
			onClick={onClick}
			aria-label={label}
			variant="ghost"
			size="icon-lg"
			className="rounded-full text-white hover:bg-white/15 hover:text-white focus-visible:ring-white active:scale-[0.96]"
		>
			{children}
		</Button>
	);
}
