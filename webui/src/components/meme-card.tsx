
import type { SearchResult } from "@/lib/api"

interface MemeCardProps {
    meme: SearchResult
    onClick: () => void
}

export function MemeCard({ meme, onClick }: MemeCardProps) {
    return (
        <button
            onClick={onClick}
            className="group relative w-full text-left overflow-hidden rounded-md bg-card border border-border hover:border-foreground/30 transition-colors"
        >
            <div className="aspect-square overflow-hidden">
                <img
                    src={meme.url}
                    alt={meme.caption || "meme"}
                    loading="lazy"
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />
            </div>

            <div className="absolute inset-0 bg-background/80 opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-3">
                <span className="text-xs text-foreground line-clamp-2">{meme.caption}</span>
            </div>
        </button>
    )
}
