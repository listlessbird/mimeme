import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { getAdminLoginUrl } from "@/lib/admin/guard";
import { createFileRoute } from "@tanstack/react-router";
import { Github } from "lucide-react";
import { z } from "zod";

const searchSchema = z.object({
	error: z.enum(["denied", "oauth"]).optional(),
});

export const Route = createFileRoute("/admin-unlock")({
	validateSearch: searchSchema,
	loader: () => getAdminLoginUrl(),
	component: UnlockPage,
});

function UnlockPage() {
	const loginUrl = Route.useLoaderData();
	const { error } = Route.useSearch();
	const errorMessage =
		error === "denied"
			? "That GitHub account does not have admin access."
			: error === "oauth"
				? "GitHub sign-in failed. Try again."
				: null;

	return (
		<div className="flex min-h-screen items-center justify-center bg-background p-4">
			<Card className="w-full max-w-sm">
				<CardHeader>
					<CardTitle>admin access</CardTitle>
					<CardDescription>Sign in with an allowed GitHub account to continue.</CardDescription>
				</CardHeader>
				<CardContent>
					<div className="flex flex-col gap-4">
						{errorMessage ? <p className="text-xs text-destructive">{errorMessage}</p> : null}
						<Button render={<a href={loginUrl} aria-label="Continue with GitHub" />}>
							<Github />
							Continue with GitHub
						</Button>
					</div>
				</CardContent>
			</Card>
		</div>
	);
}
