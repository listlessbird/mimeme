import { DitheredLogo, type DitheredLogoProps } from "@/components/dithered-logo";
import { Leva, useControls } from "leva";

export default function DitheredLogoTweaker({
	imageSrc,
	className,
	gridSize = 280,
	scale = 1,
	dotScale = 1,
	invert = true,
	cornerRadius = 0.2,
	threshold = 180,
	contrast = 0,
	gamma = 1,
	blur = 3.75,
	diffusionStrength = 1,
	serpentine = true,
}: DitheredLogoProps) {
	const config = useControls("DitheredLogo", {
		gridSize: { value: gridSize, min: 40, max: 500, step: 10 },
		scale: { value: scale, min: 0.1, max: 2, step: 0.01 },
		dotScale: { value: dotScale, min: 0.1, max: 2, step: 0.01 },
		invert,
		cornerRadius: { value: cornerRadius, min: 0, max: 0.5, step: 0.01 },
		threshold: { value: threshold, min: 0, max: 255, step: 1 },
		contrast: { value: contrast, min: -100, max: 100, step: 1 },
		gamma: { value: gamma, min: 0.1, max: 3, step: 0.05 },
		blur: { value: blur, min: 0, max: 15, step: 0.25 },
		diffusionStrength: { value: diffusionStrength, min: 0, max: 1.5, step: 0.05 },
		serpentine,
	});

	return (
		<>
			<Leva titleBar={{ title: "DitheredLogo (dev)" }} />
			<DitheredLogo imageSrc={imageSrc} className={className} {...config} />
		</>
	);
}
