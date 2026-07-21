import { DitheredLogo } from "@/components/dithered-logo";
import { SearchBar } from "@/components/search-bar";
import { useIsMobile } from "@/hooks/use-mobile";

export default function Home() {
	const isMobile = useIsMobile();

	return (
		<div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-background p-4">
			<DitheredLogo
				imageSrc="/clown.png"
				className="aspect-square w-full max-w-85 text-foreground sm:aspect-auto sm:h-[clamp(400px,68vh,760px)] sm:max-w-4xl"
				gridSize={280}
				scale={isMobile ? 0.92 : 1.05}
				dotScale={0.8}
				invert={true}
				cornerRadius={0.2}
				gamma={1.0}
				blur={3.75}
				diffusionStrength={1.0}
			/>
			<div className="w-full max-w-xl">
				<SearchBar />
			</div>
		</div>
	);
}
