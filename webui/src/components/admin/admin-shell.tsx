import { Separator } from "@/components/ui/separator";
import {
	Sidebar,
	SidebarContent,
	SidebarGroup,
	SidebarGroupContent,
	SidebarGroupLabel,
	SidebarHeader,
	SidebarInset,
	SidebarMenu,
	SidebarMenuButton,
	SidebarMenuItem,
	SidebarProvider,
	SidebarTrigger,
} from "@/components/ui/sidebar";
import { Link, useLocation } from "@tanstack/react-router";
import { Boxes, ImageIcon, ShieldAlert } from "lucide-react";
import type { ReactNode } from "react";

export function AdminShell({ children }: { children: ReactNode }) {
	const location = useLocation();
	const onSources = location.pathname.startsWith("/admin/sources");
	const onImages = location.pathname.startsWith("/admin/images");

	return (
		<SidebarProvider>
			<Sidebar>
				<SidebarHeader>
					<div className="flex items-center gap-2 px-2 py-1.5">
						<Boxes className="size-5" />
						<span className="font-semibold">mimeme admin</span>
					</div>
				</SidebarHeader>
				<SidebarContent>
					<SidebarGroup>
						<SidebarGroupLabel>manage</SidebarGroupLabel>
						<SidebarGroupContent>
							<SidebarMenu>
								<SidebarMenuItem>
									<SidebarMenuButton asChild isActive={onSources}>
										<Link to="/admin/sources">
											<Boxes />
											<span>sources</span>
										</Link>
									</SidebarMenuButton>
								</SidebarMenuItem>
								<SidebarMenuItem>
									<SidebarMenuButton asChild isActive={onImages}>
										<Link to="/admin/images">
											<ImageIcon />
											<span>images</span>
										</Link>
									</SidebarMenuButton>
								</SidebarMenuItem>
							</SidebarMenu>
						</SidebarGroupContent>
					</SidebarGroup>
				</SidebarContent>
			</Sidebar>
			<SidebarInset>
				<header className="flex h-14 shrink-0 items-center gap-2 border-b px-4">
					<SidebarTrigger />
					<Separator orientation="vertical" className="h-4" />
					<div className="flex items-center gap-2 text-xs text-muted-foreground">
						<ShieldAlert className="size-4" />
					</div>
				</header>
				<main className="flex-1 p-4 md:p-6">{children}</main>
			</SidebarInset>
		</SidebarProvider>
	);
}
