
import { useEffect, useCallback } from "react"
import { X, Download, Copy, Link } from "lucide-react"
import type { SearchResult } from "@/lib/api"

interface MemeModalProps {
    meme: SearchResult | null
    isOpen: boolean
    onClose: () => void
}

export function MemeModal({ meme, isOpen, onClose }: MemeModalProps) {
    const handleKeyDown = useCallback(
        (e: KeyboardEvent) => {
            if (e.key === "Escape") {
                onClose()
            }
        },
        [onClose],
    )

    useEffect(() => {
        if (isOpen) {
            document.addEventListener("keydown", handleKeyDown)
            document.body.style.overflow = "hidden"
        }
        return () => {
            document.removeEventListener("keydown", handleKeyDown)
            document.body.style.overflow = "unset"
        }
    }, [isOpen, handleKeyDown])

    if (!meme || !isOpen) return null

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div className="absolute inset-0 bg-background/90 backdrop-blur-sm" onClick={onClose} />

            <div className="relative z-10 w-full max-w-2xl bg-card border border-border rounded-md overflow-hidden">
                <div className="flex items-center justify-between p-4 border-b border-border">
                    <span className="text-sm text-foreground truncate pr-4">{meme.caption || "untitled"}</span>
                    <button onClick={onClose} className="text-muted-foreground hover:text-foreground transition-colors">
                        <X className="w-4 h-4" />
                    </button>
                </div>

                <div className="p-4">
                    <img
                        src={meme.url}
                        alt={meme.caption || "meme"}
                        className="w-full max-h-[60vh] object-contain rounded-sm"
                    />
                </div>

                {meme.ocr_text && (
                    <div className="px-4 pb-2">
                        <p className="text-xs text-muted-foreground italic">"{meme.ocr_text}"</p>
                    </div>
                )}

                <div className="flex items-center justify-between p-4 border-t border-border">
                    <span className="text-xs text-muted-foreground">
                        score: {meme.score.toFixed(3)}
                    </span>

                    <div className="flex items-center gap-3">
                        <button className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1.5 transition-colors">
                            <Copy className="w-3 h-3" />
                            copy
                        </button>
                        <button className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1.5 transition-colors">
                            <Link className="w-3 h-3" />
                            link
                        </button>
                        <button className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1.5 transition-colors">
                            <Download className="w-3 h-3" />
                            save
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )
}
