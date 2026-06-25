import { SearchBar } from "@/components/search-bar";

export default function Home() {
	return (
		<div className="flex min-h-screen items-center justify-center bg-background p-4">
			<div className="w-full max-w-xl">
				<SearchBar />
			</div>
		</div>
	);
}
