import * as echarts from "echarts";
import { useEffect, useRef } from "react";
import { useApp } from "./context";

export const COLORS = {
  light: { actual: "#1d4ed8", forecast: "#c2410c", baseline: "#6b7280", up: "#15803d", down: "#b91c1c",
           grid: "#e5e7eb", text: "#374151", muted: "#6b7280", band: "rgba(194,65,12,0.14)", bg: "transparent" },
  dark: { actual: "#60a5fa", forecast: "#fb923c", baseline: "#9ca3af", up: "#4ade80", down: "#f87171",
          grid: "#2a3341", text: "#d1d5db", muted: "#9ca3af", band: "rgba(251,146,60,0.18)", bg: "transparent" },
};

export type Palette = typeof COLORS.light;

export function usePalette(): Palette {
  const { theme } = useApp();
  return COLORS[theme];
}

/** Common axis/tooltip styling so every chart reads the same in both themes. */
export function baseOption(p: Palette): echarts.EChartsOption {
  return {
    backgroundColor: p.bg,
    textStyle: { color: p.text, fontFamily: "inherit" },
    grid: { left: 56, right: 18, top: 36, bottom: 56, containLabel: false },
    tooltip: { trigger: "axis", confine: true, axisPointer: { type: "cross", label: { backgroundColor: p.muted } } },
    legend: { top: 0, textStyle: { color: p.text } },
  };
}

export function axisStyle(p: Palette) {
  return {
    axisLine: { lineStyle: { color: p.grid } },
    axisLabel: { color: p.muted },
    splitLine: { lineStyle: { color: p.grid } },
  };
}

/**
 * Thin React wrapper around ECharts. Charts sharing a `group` have synchronised tooltips,
 * axis pointers and dataZoom (echarts.connect).
 */
export function EChart({ option, height = 360, group, ariaLabel, onReady }: {
  option: echarts.EChartsOption; height?: number | string; group?: string; ariaLabel?: string;
  onReady?: (c: echarts.ECharts) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const c = echarts.init(ref.current, undefined, { renderer: "canvas" });
    chart.current = c;
    if (group) {
      c.group = group;
      echarts.connect(group);
    }
    onReady?.(c);
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(ref.current);
    return () => {
      ro.disconnect();
      c.dispose();
      chart.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [group]);

  useEffect(() => {
    chart.current?.setOption(option, { notMerge: true });
  }, [option]);

  return <div ref={ref} className="echart" style={{ height }} role="img" aria-label={ariaLabel} />;
}
