
import { useState } from "react"
import { motion, useReducedMotion } from "motion/react"
import type { SearchResult } from "@/lib/api"

interface MemeCardProps {
    meme: SearchResult
    onClick: () => void
}

const easeOutQuint = [0.23, 1, 0.32, 1] as const

export function MemeCard({ meme, onClick }: MemeCardProps) {
    const [loaded, setLoaded] = useState(false)
    const shouldReduceMotion = useReducedMotion()

    return (
        <motion.button
            onClick={onClick}
            className="group relative w-full text-left overflow-hidden rounded-md bg-card border border-border hover:border-foreground/30 transition-colors will-change-transform"
            variants={{
                hidden: shouldReduceMotion
                    ? { opacity: 1, y: 0 }
                    : { opacity: 0, y: 12 },
                show: { opacity: 1, y: 0 },
            }}
            transition={{ duration: 0.2, ease: easeOutQuint }}
            whileHover={shouldReduceMotion ? undefined : { scale: 1.02 }}
            whileTap={shouldReduceMotion ? undefined : { scale: 0.97 }}
        >
            <div className="aspect-square overflow-hidden relative">
                {!loaded && (
                    <div className="absolute inset-0 shimmer rounded-sm" />
                )}
                <img
                    src={meme.url}
                    alt={meme.caption || "meme"}
                    loading="lazy"
                    onLoad={() => setLoaded(true)}
                    className={`w-full h-full object-cover transition-all duration-500 ease-out ${
                        loaded ? "opacity-100 scale-100" : "opacity-0 scale-[1.03]"
                    }`}
                    style={shouldReduceMotion ? { opacity: 1, transform: "none", transition: "none" } : undefined}
                />
            </div>

            <div className="absolute inset-0 bg-background/80 opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-3">
                <span className="text-xs text-foreground line-clamp-2">{meme.caption}</span>
            </div>
        </motion.button>
    )
}
