// Small, documented client-side computations on genuine observed data.

/** Drawdown from running peak of a price series: close / max(close so far) - 1. */
export function drawdown(closes: number[]): number[] {
  let peak = -Infinity;
  return closes.map((c) => {
    peak = Math.max(peak, c);
    return c / peak - 1;
  });
}

/** Trailing annualised realised volatility of log returns over `window` observations. */
export function rollingVol(logReturns: (number | null)[], window = 21): (number | null)[] {
  return logReturns.map((_, i) => {
    if (i + 1 < window) return null;
    const w = logReturns.slice(i + 1 - window, i + 1).filter((x): x is number => x !== null);
    if (w.length < window * 0.8) return null;
    const m = w.reduce((a, b) => a + b, 0) / w.length;
    const v = w.reduce((a, b) => a + (b - m) ** 2, 0) / (w.length - 1);
    return Math.sqrt(v * 252);
  });
}

/** Simple moving average (null until the window is full). */
export function sma(values: number[], window: number): (number | null)[] {
  let sum = 0;
  return values.map((v, i) => {
    sum += v;
    if (i >= window) sum -= values[i - window];
    return i + 1 >= window ? sum / window : null;
  });
}

export function histogram(values: number[], bins = 30): { x0: number; x1: number; count: number }[] {
  const v = values.filter(Number.isFinite);
  if (!v.length) return [];
  const lo = Math.min(...v);
  const hi = Math.max(...v);
  const w = (hi - lo) / bins || 1;
  const out = Array.from({ length: bins }, (_, i) => ({ x0: lo + i * w, x1: lo + (i + 1) * w, count: 0 }));
  v.forEach((x) => {
    out[Math.min(bins - 1, Math.floor((x - lo) / w))].count += 1;
  });
  return out;
}

/** Percentile rank (0-1) of `x` within `sample`. */
export function percentileRank(sample: number[], x: number): number {
  const s = sample.filter(Number.isFinite);
  if (!s.length) return NaN;
  return s.filter((v) => v <= x).length / s.length;
}

export function mean(v: number[]): number {
  const s = v.filter(Number.isFinite);
  return s.length ? s.reduce((a, b) => a + b, 0) / s.length : NaN;
}
