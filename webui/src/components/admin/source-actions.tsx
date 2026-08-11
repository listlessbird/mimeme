import {
	AlertDialog,
	AlertDialogAction,
	AlertDialogCancel,
	AlertDialogContent,
	AlertDialogDescription,
	AlertDialogFooter,
	AlertDialogHeader,
	AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
	adminErrorMessage as errorMessage,
	adminQueryKeys,
	deleteSource,
	type SourceDetail,
	triggerRun,
	updateSource,
} from "@/lib/admin/api";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { Pencil, Play, Power, RotateCcw, Trash2 } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { SourceFormSheet } from "./source-form-sheet";
import { useRetrySource } from "./use-source-retry";

export function SourceActions({ source }: { source: SourceDetail }) {
	const queryClient = useQueryClient();
	const navigate = useNavigate();
	const [editOpen, setEditOpen] = useState(false);
	const [deleteOpen, setDeleteOpen] = useState(false);

	const invalidate = () =>
		Promise.all([
			queryClient.invalidateQueries({ queryKey: adminQueryKeys.source(source.id) }),
			queryClient.invalidateQueries({ queryKey: adminQueryKeys.sources }),
		]);

	const run = useMutation({
		mutationFn: () => triggerRun({ data: { id: source.id } }),
		onSuccess: async () => {
			toast.success("run started", {
				description: "it will appear below as polling settles its counts.",
			});
			await invalidate();
		},
		onError: (error) => toast.error(errorMessage(error, "could not start a run")),
	});

	const retry = useRetrySource(source.id);
	const failedCount = source.stats.failed_count;

	const toggle = useMutation({
		mutationFn: () =>
			updateSource({
				data: {
					id: source.id,
					body: {
						adapter_config: {
							subreddits: source.adapter_config?.subreddits ?? [],
						},
						dataset: source.dataset,
						schedule_cron: source.schedule_cron,
						schedule_timezone: source.schedule_timezone,
						max_items_per_run: source.max_items_per_run,
						enabled: !source.enabled,
					},
				},
			}),
		onSuccess: async () => {
			toast.success(source.enabled ? "source disabled" : "source enabled");
			await invalidate();
		},
		onError: (error) => toast.error(errorMessage(error, "could not update the source")),
	});

	const remove = useMutation({
		mutationFn: () => deleteSource({ data: { id: source.id } }),
		onSuccess: async () => {
			toast.success("source deleted");
			await queryClient.invalidateQueries({ queryKey: adminQueryKeys.sources });
			await navigate({ to: "/admin/sources" });
		},
		onError: (error) => toast.error(errorMessage(error, "could not delete the source")),
	});

	return (
		<div className="flex flex-wrap items-center gap-2">
			<Button onClick={() => run.mutate()} disabled={run.isPending}>
				{run.isPending ? <Spinner data-icon="inline-start" /> : <Play data-icon="inline-start" />}
				run now
			</Button>

			{failedCount > 0 ? (
				<Button variant="outline" onClick={() => retry.mutate()} disabled={retry.isPending}>
					{retry.isPending ? (
						<Spinner data-icon="inline-start" />
					) : (
						<RotateCcw data-icon="inline-start" />
					)}
					retry {failedCount} failed
				</Button>
			) : null}

			<Button variant="outline" onClick={() => toggle.mutate()} disabled={toggle.isPending}>
				<Power data-icon="inline-start" />
				{source.enabled ? "disable" : "enable"}
			</Button>

			<Button variant="outline" onClick={() => setEditOpen(true)}>
				<Pencil data-icon="inline-start" />
				edit
			</Button>

			<Button variant="outline" onClick={() => setDeleteOpen(true)}>
				<Trash2 data-icon="inline-start" />
				delete
			</Button>

			<SourceFormSheet mode="edit" source={source} open={editOpen} onOpenChange={setEditOpen} />

			<AlertDialog open={deleteOpen} onOpenChange={setDeleteOpen}>
				<AlertDialogContent>
					<AlertDialogHeader>
						<AlertDialogTitle>delete this source?</AlertDialogTitle>
						<AlertDialogDescription>
							“{source.name}” will be soft-deleted and removed from the list. Its run and discovery
							history are retained. This cannot be undone from the UI.
						</AlertDialogDescription>
					</AlertDialogHeader>
					<AlertDialogFooter>
						<AlertDialogCancel>cancel</AlertDialogCancel>
						<AlertDialogAction onClick={() => remove.mutate()} disabled={remove.isPending}>
							{remove.isPending ? <Spinner data-icon="inline-start" /> : null}
							delete
						</AlertDialogAction>
					</AlertDialogFooter>
				</AlertDialogContent>
			</AlertDialog>
		</div>
	);
}
