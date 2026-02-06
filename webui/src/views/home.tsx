
import { SearchBar } from "@/components/search-bar"

export default function Home() {
    return (
        <div className="min-h-screen bg-background flex items-center justify-center p-4">
            <div className="w-full max-w-xl">
                <SearchBar />
            </div>
        </div>
    )
}
