
import { useEffect, useCallback, useState } from "react"
import { motion, useReducedMotion } from "motion/react"
import { X, Download, Copy, Check } from "lucide-react"
import type { SearchResult } from "@/lib/api"

interface MemeModalProps {
    meme: SearchResult
    onClose: () => void
}

const easeOutQuint = [0.23, 1, 0.32, 1] as const

export function MemeModal({ meme, onClose }: MemeModalProps) {
    const shouldReduceMotion = useReducedMotion()
    const [copied, setCopied] = useState(false)

    const handleKeyDown = useCallback(
        (e: KeyboardEvent) => {
            if (e.key === "Escape") onClose()
        },
        [onClose],
    )

    useEffect(() => {
        document.addEventListener("keydown", handleKeyDown)
        document.body.style.overflow = "hidden"
        return () => {
            document.removeEventListener("keydown", handleKeyDown)
            document.body.style.overflow = "unset"
        }
    }, [handleKeyDown])

    const copyToClipboard = async (text: string) => {
        await navigator.clipboard.writeText(text)
        setCopied(true)
        setTimeout(() => setCopied(false), 1500)
    }

    return (
        <motion.div
            className="fixed inset-0 z-50 flex items-end sm:items-center justify-center"
            initial={shouldReduceMotion ? false : { opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, transition: { duration: 0.15, ease: "easeIn" } }}
            transition={{ duration: 0.2, ease: easeOutQuint }}
        >
            <motion.div
                className="absolute inset-0 bg-background/90 backdrop-blur-sm"
                onClick={onClose}
                initial={shouldReduceMotion ? false : { opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0, transition: { duration: 0.15, ease: "easeIn" } }}
                transition={{ duration: 0.2, ease: easeOutQuint }}
            />

            <motion.div
                className="relative z-10 w-full max-w-2xl bg-card border border-border rounded-t-lg sm:rounded-lg overflow-hidden will-change-transform"
                initial={shouldReduceMotion ? false : { opacity: 0, scale: 0.95, y: 20 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.97, y: 10, transition: { duration: 0.15, ease: "easeIn" } }}
                transition={{ duration: 0.2, ease: easeOutQuint }}
            >
                {/* Header — close only */}
                <div className="flex items-center justify-end p-4 border-b border-border">
                    <button onClick={onClose} className="p-2 -mr-2 text-muted-foreground hover:text-foreground transition-colors">
                        <X className="w-4 h-4" />
                    </button>
                </div>

                {/* Image */}
                <div className="p-4">
                    <img
                        src={meme.url}
                        alt={meme.caption || "meme"}
                        className="w-full max-h-[60vh] object-contain rounded-sm"
                    />
                </div>

                {/* Info — bottom panel */}
                <div className="px-4 pt-3 pb-4 space-y-2">
                    {meme.caption && (
                        <p className="text-sm text-foreground leading-relaxed">{meme.caption}</p>
                    )}

                    {meme.ocr_text && (
                        <p className="text-xs text-muted-foreground italic leading-relaxed">"{meme.ocr_text}"</p>
                    )}

                    <div className="flex items-center justify-between pt-2 border-t border-border">
                        <span className="text-[10px] text-muted-foreground/60 tabular-nums tracking-wider uppercase">
                            {meme.score.toFixed(3)} · {meme.width}×{meme.height}
                        </span>

                        <div className="flex items-center gap-1">
                            <ActionButton
                                icon={copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                                label={copied ? "copied" : "copy"}
                                onClick={() => copyToClipboard(meme.url)}
                            />
                            <ActionButton
                                icon={<Download className="w-3 h-3" />}
                                label="save"
                                onClick={() => window.open(meme.url, "_blank")}
                            />
                        </div>
                    </div>
                </div>
            </motion.div>
        </motion.div>
    )
}

function ActionButton({ icon, label, onClick }: { icon: React.ReactNode; label: string; onClick: () => void }) {
    return (
        <button
            onClick={onClick}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs text-muted-foreground hover:text-foreground hover:bg-secondary/80 transition-colors"
        >
            {icon}
            {label}
        </button>
    )
}
