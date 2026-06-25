import { Button } from "@/components/ui/button";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
	DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
	AdminApiError,
	type ImageIngestResponse,
	ingestImageUrls,
	uploadImageFile,
} from "@/lib/admin/api";
import { cn } from "@/lib/utils";
import { useQueryClient } from "@tanstack/react-query";
import { Plus, Upload, X } from "lucide-react";
import { useRef, useState } from "react";
import { toast } from "sonner";

import { JobIdChip } from "./job-detail";

interface IngestResult {
	label: string;
	response: ImageIngestResponse;
}

function parseTags(raw: string): string[] {
	return raw
		.split(",")
		.map((t) => t.trim())
		.filter(Boolean);
}

function parseUrls(raw: string): string[] {
	return raw
		.split(/\s+/)
		.map((u) => u.trim())
		.filter(Boolean);
}

export function ImageIngestDialog() {
	const queryClient = useQueryClient();
	const [open, setOpen] = useState(false);
	const [dataset, setDataset] = useState("");
	const [tags, setTags] = useState("");
	const [urls, setUrls] = useState("");
	const [files, setFiles] = useState<File[]>([]);
	const [dragging, setDragging] = useState(false);
	const [submitting, setSubmitting] = useState(false);
	const [results, setResults] = useState<IngestResult[]>([]);
	const inputRef = useRef<HTMLInputElement>(null);

	function addFiles(list: FileList | null) {
		if (!list) return;
		setFiles((prev) => [...prev, ...Array.from(list)]);
	}

	async function submit() {
		const urlList = parseUrls(urls);
		if (urlList.length === 0 && files.length === 0) {
			toast.error("add at least one url or file");
			return;
		}

		const tagList = parseTags(tags);
		const ds = dataset.trim() || null;
		setSubmitting(true);
		const queued: IngestResult[] = [];

		try {
			if (urlList.length > 0) {
				const response = await ingestImageUrls({
					data: { urls: urlList, dataset: ds, tags: tagList },
				});
				queued.push({ label: `${urlList.length} url${urlList.length > 1 ? "s" : ""}`, response });
			}

			for (const file of files) {
				const form = new FormData();
				form.append("file", file);
				if (ds) form.append("dataset", ds);
				for (const tag of tagList) form.append("tags", tag);
				const response = await uploadImageFile({ data: form });
				queued.push({ label: file.name, response });
			}

			setResults((prev) => [...queued, ...prev]);
			setUrls("");
			setFiles([]);
			await queryClient.invalidateQueries({ queryKey: ["admin", "images"] });
			toast.success("queued for ingest");
		} catch (error) {
			toast.error(error instanceof AdminApiError ? error.message : "ingest failed");
		} finally {
			setSubmitting(false);
		}
	}

	return (
		<Dialog open={open} onOpenChange={setOpen}>
			<DialogTrigger asChild>
				<Button>
					<Plus data-icon="inline-start" />
					ingest
				</Button>
			</DialogTrigger>
			<DialogContent className="max-w-lg gap-4">
				<DialogHeader>
					<DialogTitle>ingest images</DialogTitle>
					<DialogDescription>
						paste image urls or drop files. both converge on the same ingest + dedup path.
					</DialogDescription>
				</DialogHeader>

				<div className="grid grid-cols-2 gap-3">
					<div className="flex flex-col gap-1.5">
						<Label htmlFor="ingest-dataset">dataset</Label>
						<Input
							id="ingest-dataset"
							value={dataset}
							placeholder="optional"
							onChange={(e) => setDataset(e.target.value)}
						/>
					</div>
					<div className="flex flex-col gap-1.5">
						<Label htmlFor="ingest-tags">tags</Label>
						<Input
							id="ingest-tags"
							value={tags}
							placeholder="comma, separated"
							onChange={(e) => setTags(e.target.value)}
						/>
					</div>
				</div>

				<div className="flex flex-col gap-1.5">
					<Label htmlFor="ingest-urls">urls</Label>
					<Textarea
						id="ingest-urls"
						value={urls}
						placeholder="one url per line"
						rows={3}
						onChange={(e) => setUrls(e.target.value)}
					/>
				</div>

				<button
					type="button"
					onClick={() => inputRef.current?.click()}
					onDragOver={(e) => {
						e.preventDefault();
						setDragging(true);
					}}
					onDragLeave={() => setDragging(false)}
					onDrop={(e) => {
						e.preventDefault();
						setDragging(false);
						addFiles(e.dataTransfer.files);
					}}
					className={cn(
						"flex flex-col items-center justify-center gap-1 rounded-md border border-dashed p-6 text-sm text-muted-foreground transition-colors",
						dragging ? "border-ring bg-accent/40" : "hover:bg-accent/30",
					)}
				>
					<Upload className="size-5" />
					drop files or click to choose
					<input
						ref={inputRef}
						type="file"
						accept="image/*"
						multiple
						className="hidden"
						onChange={(e) => {
							addFiles(e.target.files);
							e.target.value = "";
						}}
					/>
				</button>

				{files.length > 0 ? (
					<div className="flex flex-wrap gap-1.5">
						{files.map((file, i) => (
							<span
								key={`${file.name}-${i}`}
								className="inline-flex items-center gap-1 rounded-md border bg-muted px-2 py-0.5 text-xs"
							>
								<span className="max-w-[16ch] truncate">{file.name}</span>
								<button
									type="button"
									aria-label={`remove ${file.name}`}
									onClick={() => setFiles((prev) => prev.filter((_, idx) => idx !== i))}
								>
									<X className="size-3" />
								</button>
							</span>
						))}
					</div>
				) : null}

				{results.length > 0 ? (
					<div className="flex flex-col gap-2 rounded-md border p-3">
						<p className="text-xs font-medium text-muted-foreground">queued jobs</p>
						{results.map((result) => (
							<div
								key={result.response.job_id}
								className="flex flex-wrap items-center gap-2 text-sm"
							>
								<span className="min-w-0 truncate text-muted-foreground">{result.label}</span>
								<span className="tabular-nums">
									{result.response.queued} queued · {result.response.duplicates} deduped
								</span>
								<JobIdChip jobId={result.response.job_id} />
							</div>
						))}
					</div>
				) : null}

				<DialogFooter>
					<Button onClick={submit} disabled={submitting}>
						{submitting ? "queuing…" : "queue ingest"}
					</Button>
				</DialogFooter>
			</DialogContent>
		</Dialog>
	);
}
