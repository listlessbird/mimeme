import { Button } from "@/components/ui/button";
import {
	Drawer,
	DrawerContent,
	DrawerDescription,
	DrawerHeader,
	DrawerTitle,
	DrawerTrigger,
} from "@/components/ui/drawer";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Braces } from "lucide-react";
import { type ReactNode, useMemo } from "react";

import { CopyButton } from "./copy-button";

interface RawJsonDrawerProps {
	data: unknown;
	title?: string;
	description?: string;
	trigger?: ReactNode;
}

export function RawJsonDrawer({
	data,
	title = "raw json",
	description,
	trigger,
}: RawJsonDrawerProps) {
	const json = useMemo(() => JSON.stringify(data, null, 2), [data]);

	return (
		<Drawer>
			<DrawerTrigger asChild>
				{trigger ?? (
					<Button type="button" variant="outline" size="xs">
						<Braces />
						raw json
					</Button>
				)}
			</DrawerTrigger>
			<DrawerContent>
				<div className="mx-auto flex w-full max-w-3xl flex-col overflow-hidden">
					<DrawerHeader className="flex-row items-start justify-between gap-4">
						<div className="flex flex-col gap-1 text-left">
							<DrawerTitle className="font-mono text-sm">{title}</DrawerTitle>
							<DrawerDescription>
								{description ?? "the exact payload behind this view"}
							</DrawerDescription>
						</div>
						<CopyButton value={json} label="json" variant="outline">
							copy all
						</CopyButton>
					</DrawerHeader>
					<ScrollArea className="max-h-[60vh] px-4 pb-6">
						<pre className="rounded-md bg-muted p-4 font-mono text-xs leading-relaxed break-all whitespace-pre-wrap">
							{json}
						</pre>
					</ScrollArea>
				</div>
			</DrawerContent>
		</Drawer>
	);
}
