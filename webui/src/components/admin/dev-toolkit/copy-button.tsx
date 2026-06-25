import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Check, Copy } from "lucide-react";
import { type ReactNode, useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";

export function useCopy(resetMs = 1500) {
	const [copied, setCopied] = useState(false);
	const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

	useEffect(() => () => clearTimeout(timer.current ?? undefined), []);

	const copy = useCallback(
		async (value: string, label?: string) => {
			try {
				await navigator.clipboard.writeText(value);
				setCopied(true);
				toast.success(label ? `copied ${label}` : "copied to clipboard");
				clearTimeout(timer.current ?? undefined);
				timer.current = setTimeout(() => setCopied(false), resetMs);
			} catch {
				toast.error("could not copy to clipboard");
			}
		},
		[resetMs],
	);

	return { copied, copy };
}

interface CopyButtonProps {
	value: string;
	label?: string;
	className?: string;
	variant?: "ghost" | "outline" | "secondary";
	children?: ReactNode;
}

export function CopyButton({
	value,
	label,
	className,
	variant = "ghost",
	children,
}: CopyButtonProps) {
	const { copied, copy } = useCopy();

	return (
		<Button
			type="button"
			variant={variant}
			size={children ? "xs" : "icon-xs"}
			className={cn(className)}
			aria-label={`copy ${label ?? "value"}`}
			onClick={(e) => {
				e.stopPropagation();
				void copy(value, label);
			}}
		>
			{copied ? <Check /> : <Copy />}
			{children}
		</Button>
	);
}
