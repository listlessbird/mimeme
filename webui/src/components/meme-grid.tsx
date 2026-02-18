
import { useState } from "react"
import { motion, AnimatePresence, useReducedMotion } from "motion/react"
import { MemeCard } from "@/components/meme-card"
import { MemeModal } from "@/components/meme-modal"
import type { SearchResponse, SearchResult } from "@/lib/api"

interface MemeGridProps {
    data: SearchResponse
}

export function MemeGrid({ data }: MemeGridProps) {
    const [selectedMeme, setSelectedMeme] = useState<SearchResult | null>(null)
    const shouldReduceMotion = useReducedMotion()

    if (data.results.length === 0) {
        return (
            <motion.div
                className="text-center py-20 text-muted-foreground text-sm"
                initial={shouldReduceMotion ? false : { opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25, ease: [0.23, 1, 0.32, 1] }}
            >
                <p>no results for "{data.query}"</p>
                <p className="mt-2 text-xs">try shorter phrases or different keywords</p>
            </motion.div>
        )
    }

    return (
        <>
            <motion.div
                className="mb-4 text-xs text-muted-foreground"
                initial={shouldReduceMotion ? false : { opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.25 }}
            >
                <span>
                    {data.total} results for "{data.query}" ({data.search_time_ms.toFixed(0)}ms)
                </span>
            </motion.div>

            <motion.div
                className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3"
                key={data.query}
                initial="hidden"
                animate="show"
                variants={{
                    hidden: {},
                    show: {
                        transition: {
                            staggerChildren: shouldReduceMotion ? 0 : 0.04,
                        },
                    },
                }}
            >
                {data.results.map((meme) => (
                    <MemeCard key={meme.id} meme={meme} onClick={() => setSelectedMeme(meme)} />
                ))}
            </motion.div>

            <AnimatePresence>
                {selectedMeme && (
                    <MemeModal meme={selectedMeme} onClose={() => setSelectedMeme(null)} />
                )}
            </AnimatePresence>
        </>
    )
}
