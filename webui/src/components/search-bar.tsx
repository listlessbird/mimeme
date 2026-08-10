import { Button } from "@/components/ui/button";
import { Kbd } from "@/components/ui/kbd";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { Search, Loader2 } from "lucide-react";
import { useQueryState, parseAsString, debounce } from "nuqs";
import { useState, useRef, useEffect, useCallback } from "react";

function isMac() {
	return typeof navigator !== "undefined" && /mac/i.test(navigator.userAgent);
}

interface SearchBarProps {
	/** When true, syncs input with URL ?q= param via nuqs (for results page) */
	live?: boolean;
	isSearching?: boolean;
}

export function SearchBar({ live = false, isSearching = false }: SearchBarProps) {
	if (live) return <LiveSearchBar isSearching={isSearching} />;
	return <NavigateSearchBar isSearching={isSearching} />;
}

/** Used on the results page — syncs input state with ?q= in the URL */
function LiveSearchBar({ isSearching }: { isSearching: boolean }) {
	const inputRef = useRef<HTMLInputElement>(null);
	const [focused, setFocused] = useState(false);
	const [q, setQ] = useQueryState(
		"q",
		parseAsString.withDefault("").withOptions({
			shallow: false,
		}),
	);

	const handleShortcut = useCallback((e: KeyboardEvent) => {
		if ((e.ctrlKey || e.metaKey) && e.key === "k") {
			e.preventDefault();
			inputRef.current?.focus();
		}
	}, []);

	useEffect(() => {
		document.addEventListener("keydown", handleShortcut);
		return () => document.removeEventListener("keydown", handleShortcut);
	}, [handleShortcut]);

	return (
		<form
			onSubmit={(e) => {
				e.preventDefault();
				void setQ(q.trim());
			}}
			className="mb-6"
		>
			<div className="flex items-center gap-3 rounded-md border border-border bg-card px-4 py-3 transition-colors focus-within:border-foreground/50">
				{isSearching ? (
					<Loader2 className="h-4 w-4 shrink-0 animate-spin text-muted-foreground" />
				) : (
					<Search className="h-4 w-4 shrink-0 text-muted-foreground" />
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
					<div className="hidden shrink-0 items-center gap-1 sm:flex">
						<Kbd>{isMac() ? "⌘" : "Ctrl"}</Kbd>
						<Kbd>K</Kbd>
					</div>
				)}
			</div>
		</form>
	);
}

/** Used on the home page — navigates to /results?q=... on submit */
function NavigateSearchBar({ isSearching }: { isSearching: boolean }) {
	const [query, setQuery] = useState("");
	const navigate = useNavigate();
	const isRouterPending = useRouterState({ select: (s) => s.isLoading });
	const loading = isSearching || isRouterPending;

	const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
		e.preventDefault();
		if (query.trim()) {
			void navigate({ to: "/results", search: { q: query.trim() } });
		}
	};

	return (
		<form onSubmit={handleSubmit} className="mb-6">
			<div className="flex items-center gap-3 rounded-md border border-border bg-card px-4 py-3 transition-colors focus-within:border-foreground/50">
				{loading ? (
					<Loader2 className="h-4 w-4 shrink-0 animate-spin text-muted-foreground" />
				) : (
					<Search className="h-4 w-4 shrink-0 text-muted-foreground" />
				)}
				<input
					type="text"
					value={query}
					onChange={(e) => setQuery(e.target.value)}
					placeholder="search..."
					className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
					// oxlint-disable-next-line jsx-a11y/no-autofocus -- hero search is the primary action on this page
					autoFocus
				/>
				<Button
					type="submit"
					variant="ghost"
					size="sm"
					disabled={!query.trim() || loading}
					className="h-auto px-0 text-xs font-normal text-muted-foreground hover:bg-transparent hover:text-foreground disabled:opacity-40"
				>
					{loading ? "searching…" : "[enter]"}
				</Button>
			</div>
		</form>
	);
}
