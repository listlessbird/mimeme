
import type React from "react"
import { useState } from "react"
import { Search, Loader2 } from "lucide-react"
import { useNavigate } from "@tanstack/react-router"
import { useQueryState, parseAsString, debounce } from "nuqs"

interface SearchBarProps {
    /** When true, syncs input with URL ?q= param via nuqs (for results page) */
    live?: boolean
    isSearching?: boolean
}

export function SearchBar({ live = false, isSearching = false }: SearchBarProps) {
    if (live) return <LiveSearchBar isSearching={isSearching} />
    return <NavigateSearchBar isSearching={isSearching} />
}

/** Used on the results page — syncs input state with ?q= in the URL */
function LiveSearchBar({ isSearching }: { isSearching: boolean }) {
    const [q, setQ] = useQueryState(
        "q",
        parseAsString.withDefault("").withOptions({
            shallow: false,
        }),
    )

    return (
        <form
            onSubmit={(e) => {
                e.preventDefault()
                setQ(q.trim())
            }}
            className="mb-6"
        >
            <div className="flex items-center gap-3 bg-card border border-border rounded-md px-4 py-3 focus-within:border-foreground/50 transition-colors">
                {isSearching ? (
                    <Loader2 className="w-4 h-4 text-muted-foreground animate-spin shrink-0" />
                ) : (
                    <Search className="w-4 h-4 text-muted-foreground shrink-0" />
                )}
                <input
                    type="text"
                    value={q}
                    onChange={(e) =>
                        setQ(e.target.value, {
                            limitUrlUpdates: e.target.value === "" ? undefined : debounce(300),
                        })
                    }
                    placeholder="describe the meme you're looking for..."
                    className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
                />
            </div>
        </form>
    )
}

/** Used on the home page — navigates to /results?q=... on submit */
function NavigateSearchBar({ isSearching }: { isSearching: boolean }) {
    const [query, setQuery] = useState("")
    const navigate = useNavigate()

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault()
        if (query.trim()) {
            navigate({ to: "/results", search: { q: query.trim() } })
        }
    }

    return (
        <form onSubmit={handleSubmit} className="mb-6">
            <div className="flex items-center gap-3 bg-card border border-border rounded-md px-4 py-3 focus-within:border-foreground/50 transition-colors">
                {isSearching ? (
                    <Loader2 className="w-4 h-4 text-muted-foreground animate-spin shrink-0" />
                ) : (
                    <Search className="w-4 h-4 text-muted-foreground shrink-0" />
                )}
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="describe the meme you're looking for..."
                    className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
                />
                <button
                    type="submit"
                    disabled={!query.trim() || isSearching}
                    className="text-xs text-muted-foreground hover:text-foreground disabled:opacity-40 transition-colors"
                >
                    [enter]
                </button>
            </div>
        </form>
    )
}
