import type { Contradiction, ContradictionType } from "./types";

export function countByType(items: Contradiction[], type: ContradictionType): number {
  return items.filter((item) => item.type === type).length;
}

export function labelForFilter(filter: ContradictionType | "ALL"): string {
  if (filter === "FALSE_POSITIVE") {
    return "False positive";
  }

  if (filter === "ALL") {
    return "All";
  }

  return filter.charAt(0) + filter.slice(1).toLowerCase();
}
