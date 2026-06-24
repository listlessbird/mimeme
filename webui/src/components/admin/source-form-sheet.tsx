import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
	Form,
	FormControl,
	FormDescription,
	FormField,
	FormItem,
	FormLabel,
	FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import {
	Select,
	SelectContent,
	SelectGroup,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import {
	Sheet,
	SheetContent,
	SheetDescription,
	SheetFooter,
	SheetHeader,
	SheetTitle,
} from "@/components/ui/sheet";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
	adminQueryKeys,
	AdminApiError,
	type CreateSourceRequest,
	createSource,
	type Source,
	type UpdateSourceRequest,
	updateSource,
} from "@/lib/admin/api";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { X } from "lucide-react";
import { useEffect } from "react";
import { useForm } from "react-hook-form";
import { toast } from "sonner";
import { z } from "zod";

const CRON_PRESETS = [
	{ label: "hourly", value: "0 * * * *" },
	{ label: "every 6h", value: "0 */6 * * *" },
	{ label: "daily", value: "0 0 * * *" },
];

const formSchema = z.object({
	name: z.string().min(1, "name is required"),
	subreddits: z.array(z.string().min(1)).min(1, "add at least one subreddit"),
	dataset: z.string(),
	schedule_cron: z.string(),
	schedule_timezone: z.string().min(1, "timezone is required"),
	max_items_per_run: z.string().refine((v) => v === "" || (/^\d+$/.test(v) && Number(v) > 0), {
		message: "must be a positive whole number",
	}),
	enabled: z.boolean(),
});

type FormValues = z.infer<typeof formSchema>;

const EMPTY_VALUES: FormValues = {
	name: "",
	subreddits: [],
	dataset: "",
	schedule_cron: "0 * * * *",
	schedule_timezone: "UTC",
	max_items_per_run: "50",
	enabled: true,
};

function sourceToFormValues(source: Source): FormValues {
	return {
		name: source.name,
		subreddits: source.adapter_config?.subreddits ?? [],
		dataset: source.dataset ?? "",
		schedule_cron: source.schedule_cron ?? "",
		schedule_timezone: source.schedule_timezone,
		max_items_per_run: source.max_items_per_run == null ? "" : String(source.max_items_per_run),
		enabled: source.enabled,
	};
}

function toNullable(value: string): string | null {
	const trimmed = value.trim();
	return trimmed === "" ? null : trimmed;
}

function maxItems(value: string): number | null {
	return value.trim() === "" ? null : Number(value);
}

interface SourceFormSheetProps {
	mode: "create" | "edit";
	source?: Source;
	open: boolean;
	onOpenChange: (open: boolean) => void;
}

export function SourceFormSheet({ mode, source, open, onOpenChange }: SourceFormSheetProps) {
	const queryClient = useQueryClient();
	const form = useForm<FormValues>({
		resolver: zodResolver(formSchema),
		defaultValues: source ? sourceToFormValues(source) : EMPTY_VALUES,
	});

	// Re-seed the form whenever it opens (so edit always reflects the latest
	// Source, and create always starts clean).
	useEffect(() => {
		if (open) {
			form.reset(source ? sourceToFormValues(source) : EMPTY_VALUES);
		}
	}, [open, source, form]);

	const mutation = useMutation({
		mutationFn: async (values: FormValues) => {
			if (mode === "create") {
				const body: CreateSourceRequest = {
					name: values.name,
					adapter_key: "meme_api",
					adapter_config: { subreddits: values.subreddits },
					dataset: toNullable(values.dataset),
					schedule_cron: toNullable(values.schedule_cron),
					schedule_timezone: values.schedule_timezone,
					max_items_per_run: maxItems(values.max_items_per_run),
					enabled: values.enabled,
				};
				return createSource({ data: body });
			}
			if (!source) throw new Error("missing source for edit");
			const body: UpdateSourceRequest = {
				adapter_config: { subreddits: values.subreddits },
				dataset: toNullable(values.dataset),
				schedule_cron: toNullable(values.schedule_cron),
				schedule_timezone: values.schedule_timezone,
				max_items_per_run: maxItems(values.max_items_per_run),
				enabled: values.enabled,
			};
			return updateSource({ data: { id: source.id, body } });
		},
		onSuccess: async (saved) => {
			toast.success(mode === "create" ? "source created" : "source updated");
			await queryClient.invalidateQueries({ queryKey: adminQueryKeys.sources });
			await queryClient.invalidateQueries({
				queryKey: adminQueryKeys.source(saved.id),
			});
			onOpenChange(false);
		},
		onError: (error) => {
			// Surface backend errors (409 duplicate name, 400 unknown adapter, …).
			const message =
				error instanceof AdminApiError && error.detail ? error.detail : "could not save the source";
			toast.error(message);
		},
	});

	return (
		<Sheet open={open} onOpenChange={onOpenChange}>
			<SheetContent className="flex w-full flex-col gap-0 sm:max-w-lg">
				<SheetHeader>
					<SheetTitle>{mode === "create" ? "new source" : "edit source"}</SheetTitle>
					<SheetDescription>
						configure a meme_api source. subreddits drive what gets discovered each run.
					</SheetDescription>
				</SheetHeader>

				<Form {...form}>
					<form
						className="flex min-h-0 flex-1 flex-col"
						onSubmit={form.handleSubmit((values) => mutation.mutate(values))}
					>
						<div className="flex flex-1 flex-col gap-5 overflow-y-auto px-4 py-2">
							<FormItem>
								<FormLabel>adapter</FormLabel>
								<Select value="meme_api" disabled>
									<SelectTrigger>
										<SelectValue />
									</SelectTrigger>
									<SelectContent>
										<SelectGroup>
											<SelectItem value="meme_api">meme_api</SelectItem>
										</SelectGroup>
									</SelectContent>
								</Select>
								<FormDescription>
									the only adapter today; more become a form branch later.
								</FormDescription>
							</FormItem>

							<FormField
								control={form.control}
								name="name"
								render={({ field }) => (
									<FormItem>
										<FormLabel>name</FormLabel>
										<FormControl>
											<Input {...field} disabled={mode === "edit"} placeholder="r/memes hourly" />
										</FormControl>
										{mode === "edit" ? (
											<FormDescription>a source's name cannot be changed.</FormDescription>
										) : null}
										<FormMessage />
									</FormItem>
								)}
							/>

							<FormField
								control={form.control}
								name="subreddits"
								render={({ field }) => (
									<FormItem>
										<FormLabel>subreddits</FormLabel>
										<FormControl>
											<SubredditInput value={field.value} onChange={field.onChange} />
										</FormControl>
										<FormMessage />
									</FormItem>
								)}
							/>

							<FormField
								control={form.control}
								name="schedule_cron"
								render={({ field }) => (
									<FormItem>
										<FormLabel>schedule (cron)</FormLabel>
										<ToggleGroup
											type="single"
											variant="outline"
											size="sm"
											value={field.value}
											onValueChange={(value) => {
												if (value) field.onChange(value);
											}}
											className="justify-start"
										>
											{CRON_PRESETS.map((preset) => (
												<ToggleGroupItem key={preset.value} value={preset.value}>
													{preset.label}
												</ToggleGroupItem>
											))}
										</ToggleGroup>
										<FormControl>
											<Input {...field} placeholder="0 * * * * (leave blank for none)" />
										</FormControl>
										<FormMessage />
									</FormItem>
								)}
							/>

							<FormField
								control={form.control}
								name="schedule_timezone"
								render={({ field }) => (
									<FormItem>
										<FormLabel>timezone</FormLabel>
										<FormControl>
											<Input {...field} placeholder="UTC" />
										</FormControl>
										<FormMessage />
									</FormItem>
								)}
							/>

							<FormField
								control={form.control}
								name="max_items_per_run"
								render={({ field }) => (
									<FormItem>
										<FormLabel>max items per run</FormLabel>
										<FormControl>
											<Input
												{...field}
												inputMode="numeric"
												placeholder="50 (blank = adapter default)"
											/>
										</FormControl>
										<FormMessage />
									</FormItem>
								)}
							/>

							<FormField
								control={form.control}
								name="dataset"
								render={({ field }) => (
									<FormItem>
										<FormLabel>dataset</FormLabel>
										<FormControl>
											<Input {...field} placeholder="memes (optional)" />
										</FormControl>
										<FormMessage />
									</FormItem>
								)}
							/>

							<FormField
								control={form.control}
								name="enabled"
								render={({ field }) => (
									<FormItem className="flex flex-row items-center justify-between gap-4 rounded-md border p-3">
										<div className="flex flex-col gap-1">
											<FormLabel>enabled</FormLabel>
											<FormDescription>
												disabled sources keep their config and dedup memory but stop acquiring.
											</FormDescription>
										</div>
										<FormControl>
											<Switch checked={field.value} onCheckedChange={field.onChange} />
										</FormControl>
									</FormItem>
								)}
							/>
						</div>

						<SheetFooter>
							<Button type="submit" disabled={mutation.isPending}>
								{mutation.isPending ? <Spinner data-icon="inline-start" /> : null}
								{mode === "create" ? "create source" : "save changes"}
							</Button>
							<Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
								cancel
							</Button>
						</SheetFooter>
					</form>
				</Form>
			</SheetContent>
		</Sheet>
	);
}

function SubredditInput({
	value,
	onChange,
}: {
	value: string[];
	onChange: (next: string[]) => void;
}) {
	function addFrom(raw: string) {
		const next = raw
			.split(",")
			.map((part) => part.trim().replace(/^r\//i, ""))
			.filter((part) => part && !value.includes(part));
		if (next.length) onChange([...value, ...next]);
	}

	return (
		<div className="flex flex-col gap-2">
			<Input
				placeholder="type a subreddit and press enter"
				onKeyDown={(event) => {
					if (event.key === "Enter" || event.key === ",") {
						event.preventDefault();
						addFrom(event.currentTarget.value);
						event.currentTarget.value = "";
					}
				}}
				onBlur={(event) => {
					addFrom(event.currentTarget.value);
					event.currentTarget.value = "";
				}}
			/>
			{value.length ? (
				<div className="flex flex-wrap gap-1.5">
					{value.map((subreddit) => (
						<Badge key={subreddit} variant="secondary" className="gap-1">
							r/{subreddit}
							<button
								type="button"
								aria-label={`remove ${subreddit}`}
								className="rounded-sm hover:text-foreground"
								onClick={() => onChange(value.filter((s) => s !== subreddit))}
							>
								<X className="size-3" />
							</button>
						</Badge>
					))}
				</div>
			) : null}
		</div>
	);
}
