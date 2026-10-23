import { describe, expect, it } from "vitest";
import { toCsv } from "./csv";
import { resolveRange, unionDates, validateRange } from "./dates";
import { logRetPct, NA, pct, taskLabel } from "./format";
import { drawdown, histogram, percentileRank, rollingVol, sma } from "./stats";

const dates = Array.from({ length: 300 }, (_, i) => {
  const d = new Date(Date.UTC(2019, 0, 1 + i));
  return d.toISOString().slice(0, 10);
});

describe("date ranges", () => {
  it("quick ranges end at the last available date and count trading days", () => {
    const r = resolveRange("30d", dates)!;
    expect(r.end).toBe(dates[299]);
    expect(dates.filter((d) => d >= r.start && d <= r.end)).toHaveLength(30);
  });
  it("month-based ranges snap to the first available date", () => {
    const r = resolveRange("6m", dates)!;
    expect(r.start >= "2019-04-27").toBe(true);
    expect(dates.includes(r.start)).toBe(true);
  });
  it("validates windows against coverage", () => {
    expect(validateRange("2018-01-01", "2018-02-01", dates).ok).toBe(false);
    expect(validateRange("2019-05-01", "2019-04-01", dates).ok).toBe(false);
    const small = validateRange("2019-01-01", "2019-01-10", dates);
    expect(small.ok).toBe(true);
    expect(small.count).toBe(10);
    expect(small.message).toMatch(/at least 20/);
    expect(validateRange("2019-01-01", "2019-06-01", dates).message).toBeUndefined();
  });
  it("union of dates is sorted and unique", () => {
    expect(unionDates(["2020-01-02", "2020-01-01"], ["2020-01-02"])).toEqual(["2020-01-01", "2020-01-02"]);
  });
});

describe("statistics", () => {
  it("drawdown from running peak", () => {
    expect(drawdown([100, 110, 99, 120])).toEqual([0, 0, 99 / 110 - 1, 0]);
  });
  it("sma is null until the window fills", () => {
    expect(sma([1, 2, 3, 4], 2)).toEqual([null, 1.5, 2.5, 3.5]);
  });
  it("rolling vol annualises the sample std", () => {
    const r = [0.01, -0.01, 0.01, -0.01];
    const v = rollingVol(r, 4);
    expect(v.slice(0, 3)).toEqual([null, null, null]);
    expect(v[3]).toBeCloseTo(Math.sqrt((4 * 0.0001) / 3 * 252), 10);
  });
  it("histogram counts every value", () => {
    expect(histogram([1, 2, 3, 4, 5], 2).reduce((a, b) => a + b.count, 0)).toBe(5);
  });
  it("percentile rank", () => {
    expect(percentileRank([1, 2, 3, 4], 3)).toBe(0.75);
  });
});

describe("formatting", () => {
  it("never formats missing values as numbers", () => {
    expect(pct(null)).toBe(NA);
    expect(pct(Number.NaN)).toBe(NA);
    expect(logRetPct(Math.log(1.1))).toBe("10.00%");
    expect(taskLabel("volatility_20")).toBe("Realised volatility · 20d");
  });
  it("csv escapes commas and quotes", () => {
    expect(toCsv([{ a: 'x,"y"', b: null }], ["a", "b"])).toBe('a,b\n"x,""y""",');
  });
});
