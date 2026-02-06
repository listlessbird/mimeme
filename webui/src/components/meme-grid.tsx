
import { useState } from "react"
import { MemeCard } from "@/components/meme-card"
import { MemeModal } from "@/components/meme-modal"
import type { SearchResponse, SearchResult } from "@/lib/api"

interface MemeGridProps {
    data: SearchResponse
}

export function MemeGrid({ data }: MemeGridProps) {
    const [selectedMeme, setSelectedMeme] = useState<SearchResult | null>(null)

    if (data.results.length === 0) {
        return (
            <div className="text-center py-20 text-muted-foreground text-sm">
                <p>no results for "{data.query}"</p>
                <p className="mt-2 text-xs">try shorter phrases or different keywords</p>
            </div>
        )
    }

    return (
        <>
            <div className="mb-4 text-xs text-muted-foreground">
                <span>
                    {data.total} results for "{data.query}" ({data.search_time_ms.toFixed(0)}ms)
                </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                {data.results.map((meme) => (
                    <MemeCard key={meme.id} meme={meme} onClick={() => setSelectedMeme(meme)} />
                ))}
            </div>

            <MemeModal meme={selectedMeme} isOpen={!!selectedMeme} onClose={() => setSelectedMeme(null)} />
        </>
    )
}
