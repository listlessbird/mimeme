import { DitheredLogo, type DitheredLogoProps } from "@/components/dithered-logo";
import { SearchBar } from "@/components/search-bar";
import { useIsMobile } from "@/hooks/use-mobile";
import { lazy, Suspense } from "react";

const LOGO_CLASS =
	"aspect-square w-full max-w-50 text-foreground sm:aspect-auto sm:h-[clamp(300px,45vh,760px)] sm:max-w-3xl";

const DitheredLogoTweaker = import.meta.env.DEV
	? lazy(() => import("@/components/dithered-logo-tweaker"))
	: null;

export default function Home() {
	const isMobile = useIsMobile();

	const logo: DitheredLogoProps = {
		imageSrc: "/clown.webp",
		className: LOGO_CLASS,
		gridSize: 390,
		scale: isMobile ? 0.92 : 0.84,
		dotScale: 0.67,
		invert: true,
		cornerRadius: 0,
		gamma: 3.0,
		blur: 4,
		diffusionStrength: 0.95,
		contrast: 100,
		threshold: 238,
	};

	return (
		<div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-background p-4">
			{DitheredLogoTweaker ? (
				<Suspense fallback={<div className={LOGO_CLASS} />}>
					<DitheredLogoTweaker {...logo} />
				</Suspense>
			) : (
				<DitheredLogo {...logo} />
			)}
			<div className="w-full max-w-xl">
				<SearchBar />
			</div>
		</div>
	);
}
