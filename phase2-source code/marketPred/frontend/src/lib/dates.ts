// Date-range helpers that work on actual trading dates from the dataset (no calendar guessing).

export type QuickRange = "30d" | "90d" | "6m" | "1y" | "all" | "custom";

export const QUICK_RANGES: { id: QuickRange; label: string }[] = [
  { id: "30d", label: "Last 30 trading days" },
  { id: "90d", label: "Last 90 trading days" },
  { id: "6m", label: "Last 6 months" },
  { id: "1y", label: "Last 1 year" },
  { id: "all", label: "Full period" },
  { id: "custom", label: "Custom" },
];

function shiftMonths(iso: string, months: number): string {
  const d = new Date(`${iso}T00:00:00Z`);
  d.setUTCMonth(d.getUTCMonth() - months);
  return d.toISOString().slice(0, 10);
}

/**
 * Resolve a quick range against the sorted list of dates that actually have evaluated predictions.
 * Ranges end at the last available date, not "today": historical evaluation can only use dates
 * whose outcomes are already observed.
 */
export function resolveRange(range: QuickRange, dates: string[]): { start: string; end: string } | null {
  if (!dates.length) return null;
  const end = dates[dates.length - 1];
  switch (range) {
    case "30d":
      return { start: dates[Math.max(0, dates.length - 30)], end };
    case "90d":
      return { start: dates[Math.max(0, dates.length - 90)], end };
    case "6m":
      return { start: firstOnOrAfter(dates, shiftMonths(end, 6)), end };
    case "1y":
      return { start: firstOnOrAfter(dates, shiftMonths(end, 12)), end };
    case "all":
      return { start: dates[0], end };
    default:
      return null;
  }
}

export function firstOnOrAfter(dates: string[], iso: string): string {
  return dates.find((d) => d >= iso) ?? dates[dates.length - 1];
}

export interface RangeCheck {
  ok: boolean;
  message?: string;
  count: number;
}

/** Validate a user-selected window against the dates that have evaluated predictions. */
export function validateRange(start: string, end: string, dates: string[], minCount = 20): RangeCheck {
  if (!dates.length) return { ok: false, count: 0, message: "No stored predictions exist for this selection." };
  if (!start || !end) return { ok: false, count: 0, message: "Choose a start and an end date." };
  if (start > end) return { ok: false, count: 0, message: "The start date is after the end date." };
  const lo = dates[0];
  const hi = dates[dates.length - 1];
  if (end < lo || start > hi) {
    return { ok: false, count: 0, message: `Predictions for this selection cover ${lo} to ${hi} only.` };
  }
  const count = dates.filter((d) => d >= start && d <= end).length;
  if (count === 0) return { ok: false, count, message: "No trading days with evaluated predictions in this window." };
  if (count < minCount) {
    return { ok: true, count, message: `Only ${count} predictions in this window; summary statistics need at least ${minCount} and are shown as N/A.` };
  }
  return { ok: true, count };
}

/** Union of dates (sorted, unique). */
export function unionDates(...lists: string[][]): string[] {
  return Array.from(new Set(lists.flat())).sort();
}
