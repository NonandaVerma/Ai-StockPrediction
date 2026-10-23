export const NA = "N/A";

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

export function num(v: unknown, digits = 4): string {
  if (!isNum(v)) return NA;
  // Values too small for fixed notation would print as a misleading 0.0000.
  return v !== 0 && Math.abs(v) < 10 ** -digits ? v.toExponential(2) : v.toFixed(digits);
}

export function pct(v: unknown, digits = 1): string {
  return isNum(v) ? `${(v * 100).toFixed(digits)}%` : NA;
}

/** Log return -> simple percentage return. */
export function logRetPct(v: unknown, digits = 2): string {
  return isNum(v) ? `${((Math.exp(v) - 1) * 100).toFixed(digits)}%` : NA;
}

export function price(v: unknown, digits = 2): string {
  return isNum(v) ? v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits }) : NA;
}

export function pval(v: unknown): string {
  if (!isNum(v)) return NA;
  return v < 0.001 ? "<0.001" : v.toFixed(3);
}

export function compact(v: unknown): string {
  if (!isNum(v)) return NA;
  return Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 2 }).format(v);
}

export const TASK_LABELS: Record<string, string> = {
  direction: "Direction (P(up))",
  return: "Return",
  volatility: "Realised volatility",
  downside: "Downside risk",
  large_move: "Large move (|1d| > 2%)",
  regime: "Market regime",
};

export function taskFamily(task: string): string {
  return task.replace(/_\d+$/, "");
}

export function taskLabel(task: string): string {
  const fam = taskFamily(task);
  const h = task.match(/_(\d+)$/)?.[1];
  return `${TASK_LABELS[fam] ?? fam}${h ? ` · ${h}d` : ""}`;
}

export const MODEL_LABELS: Record<string, string> = {
  climatology: "Climatology (base rate)",
  hist_mean: "Historical mean",
  rw_vol: "Random-walk volatility",
  har: "HAR-RV",
  garch: "GARCH(1,1)-t",
  ar1: "AR(1)",
  logit: "Logistic regression",
  ridge: "Ridge",
  enet: "Elastic net",
  lgbm: "LightGBM",
  xgb: "XGBoost",
  rf: "Random forest",
};

export const modelLabel = (m: string) => MODEL_LABELS[m] ?? m;

export function featureSetLabel(fs: string): string {
  if (fs.startsWith("combined_")) return "price + surviving categories";
  return fs.replace(/\+/g, " + ").replace(/_/g, " ");
}
