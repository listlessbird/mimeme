import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
	return twMerge(clsx(inputs));
}

export function shouldNeverHappen(message: string): never {
	throw new Error(`invariant violated: ${message}`);
}
