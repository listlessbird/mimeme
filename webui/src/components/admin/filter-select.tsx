import {
	Select,
	SelectContent,
	SelectGroup,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";


const ANY = "__any__";

export interface FilterOption {
	value: string;
	label: string;
}

interface FilterSelectProps {
	value: string;
	onValueChange: (value: string) => void;
	options: FilterOption[];
	includeAny?: boolean;
	anyLabel?: string;
	placeholder?: string;
	className?: string;
}

export function FilterSelect({
	value,
	onValueChange,
	options,
	includeAny = true,
	anyLabel = "any",
	placeholder = "any",
	className,
}: FilterSelectProps) {
	return (
		<Select
			value={value === "" ? ANY : value}
			onValueChange={(next) => onValueChange(next === ANY ? "" : next)}
		>
			<SelectTrigger className={className}>
				<SelectValue placeholder={placeholder} />
			</SelectTrigger>
			<SelectContent>
				<SelectGroup>
					{includeAny ? <SelectItem value={ANY}>{anyLabel}</SelectItem> : null}
					{options.map((option) => (
						<SelectItem key={option.value} value={option.value}>
							{option.label}
						</SelectItem>
					))}
				</SelectGroup>
			</SelectContent>
		</Select>
	);
}
