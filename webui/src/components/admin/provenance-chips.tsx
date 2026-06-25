import { IdChip } from "./dev-toolkit";

export function ImageIdChip({ imageId, label = "image_id" }: { imageId: number; label?: string }) {
	return (
		<IdChip
			label={label}
			value={imageId}
			link={{ to: "/admin/images/$imageId", params: { imageId: String(imageId) } }}
		/>
	);
}

export { JobIdChip } from "./job-detail";
