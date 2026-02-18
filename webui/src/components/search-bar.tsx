
import { useState, useRef, useEffect, useCallback } from "react"
import { Search, Loader2 } from "lucide-react"
import { useNavigate } from "@tanstack/react-router"
import { useQueryState, parseAsString, debounce } from "nuqs"
import { Kbd } from "@/components/ui/kbd"

function isMac() {
    return typeof navigator !== "undefined" && /mac/i.test(navigator.userAgent)
}

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
    const inputRef = useRef<HTMLInputElement>(null)
    const [focused, setFocused] = useState(false)
    const [q, setQ] = useQueryState(
        "q",
        parseAsString.withDefault("").withOptions({
            shallow: false,
        }),
    )

    const handleShortcut = useCallback((e: KeyboardEvent) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "k") {
            e.preventDefault()
            inputRef.current?.focus()
        }
    }, [])

    useEffect(() => {
        document.addEventListener("keydown", handleShortcut)
        return () => document.removeEventListener("keydown", handleShortcut)
    }, [handleShortcut])

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
                    ref={inputRef}
                    type="text"
                    value={q}
                    onFocus={() => setFocused(true)}
                    onBlur={() => setFocused(false)}
                    onChange={(e) =>
                        setQ(e.target.value, {
                            limitUrlUpdates: e.target.value === "" ? undefined : debounce(300),
                        })
                    }
                    placeholder="describe the meme you're looking for..."
                    className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
                />
                {!focused && (
                    <div className="hidden sm:flex items-center gap-1 shrink-0">
                        <Kbd>{isMac() ? "⌘" : "Ctrl"}</Kbd>
                        <Kbd>K</Kbd>
                    </div>
                )}
            </div>
        </form>
    )
}

/** Used on the home page — navigates to /results?q=... on submit */
function NavigateSearchBar({ isSearching }: { isSearching: boolean }) {
    const [query, setQuery] = useState("")
    const navigate = useNavigate()

    const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
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
