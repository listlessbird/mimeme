export interface GuardInputs {
	isProduction: boolean;
	secret?: string;
	cookie?: string;
}

export function isAccessAllowed({ isProduction, secret, cookie }: GuardInputs): boolean {
	if (!isProduction) return true;
	if (!secret) return false;
	return cookie === secret;
}
