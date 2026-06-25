import {
	adminErrorMessage,
	adminQueryKeys,
	retrySource,
	retrySourceItem,
	retrySourceRun,
} from "@/lib/admin/api";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

export function useRetrySource(sourceId: number) {
	const queryClient = useQueryClient();
	return useMutation({
		mutationFn: () => retrySource({ data: { id: sourceId } }),
		onSuccess: async (res) => {
			toast.success(`re-queued ${res.queued} failed ${res.queued === 1 ? "item" : "items"}`, {
				description: "watch statuses settle as the ingest reruns.",
			});
			await Promise.all([
				queryClient.invalidateQueries({ queryKey: adminQueryKeys.sourceItemsAll(sourceId) }),
				queryClient.invalidateQueries({ queryKey: adminQueryKeys.source(sourceId) }),
			]);
		},
		onError: (error) => toast.error(adminErrorMessage(error, "could not retry failed items")),
	});
}

export function useRetryRun(sourceId: number, runId: number) {
	const queryClient = useQueryClient();
	return useMutation({
		mutationFn: () => retrySourceRun({ data: { id: sourceId, runId } }),
		onSuccess: async (res) => {
			toast.success(`re-queued ${res.queued} failed ${res.queued === 1 ? "item" : "items"}`, {
				description: "watch statuses settle as the ingest reruns.",
			});
			await Promise.all([
				queryClient.invalidateQueries({ queryKey: adminQueryKeys.runItemsAll(sourceId, runId) }),
				queryClient.invalidateQueries({ queryKey: adminQueryKeys.source(sourceId) }),
			]);
		},
		onError: (error) => toast.error(adminErrorMessage(error, "could not retry this run")),
	});
}

export function useRetryItem(sourceId: number, runId?: number) {
	const queryClient = useQueryClient();
	return useMutation({
		mutationFn: (itemId: number) => retrySourceItem({ data: { id: sourceId, itemId } }),
		onSuccess: async () => {
			toast.success("re-queued item", {
				description: "watch its status settle as it reruns.",
			});
			await Promise.all([
				queryClient.invalidateQueries({ queryKey: adminQueryKeys.sourceItemsAll(sourceId) }),
				queryClient.invalidateQueries({ queryKey: adminQueryKeys.source(sourceId) }),
				...(runId === undefined
					? []
					: [
							queryClient.invalidateQueries({
								queryKey: adminQueryKeys.runItemsAll(sourceId, runId),
							}),
						]),
			]);
		},
		onError: (error) => toast.error(adminErrorMessage(error, "could not retry this item")),
	});
}
