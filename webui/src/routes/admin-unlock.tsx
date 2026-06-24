import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Field, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { unlockAdmin } from "@/lib/admin/guard";
import { useMutation } from "@tanstack/react-query";
import { createFileRoute, useRouter } from "@tanstack/react-router";
import { useState } from "react";

export const Route = createFileRoute("/admin-unlock")({
	component: UnlockPage,
});

function UnlockPage() {
	const router = useRouter();
	const [secret, setSecret] = useState("");
	const [denied, setDenied] = useState(false);

	const unlock = useMutation({
		mutationFn: (value: string) => unlockAdmin({ data: { secret: value } }),
		onSuccess: async (result) => {
			if (result.ok) {
				await router.navigate({ to: "/admin/sources" });
				return;
			}
			setDenied(true);
		},
	});

	return (
		<div className="flex min-h-screen items-center justify-center bg-background p-4">
			<Card className="w-full max-w-sm">
				<CardHeader>
					<CardTitle>admin access</CardTitle>
					<CardDescription>
						enter the shared secret to reach the acquisition admin area.
					</CardDescription>
				</CardHeader>
				<CardContent>
					<form
						className="flex flex-col gap-4"
						onSubmit={(event) => {
							event.preventDefault();
							setDenied(false);
							unlock.mutate(secret);
						}}
					>
						<Field data-invalid={denied || undefined}>
							<FieldLabel htmlFor="admin-secret">shared secret</FieldLabel>
							<Input
								id="admin-secret"
								type="password"
								autoComplete="off"
								value={secret}
								aria-invalid={denied || undefined}
								onChange={(event) => setSecret(event.target.value)}
							/>
						</Field>
						{denied ? (
							<p className="text-xs text-destructive">that secret was not accepted.</p>
						) : null}
						<Button type="submit" disabled={unlock.isPending || !secret}>
							{unlock.isPending ? <Spinner data-icon="inline-start" /> : null}
							unlock
						</Button>
					</form>
				</CardContent>
			</Card>
		</div>
	);
}
