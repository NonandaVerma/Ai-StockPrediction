import type { EChartsOption } from "echarts";
import { useMemo } from "react";
import { api } from "../api/client";
import type { AvailablePrediction, Forecast, PredictionRow, PriceRow } from "../api/types";
import { axisStyle, baseOption, EChart, Palette, usePalette } from "../components/EChart";
import { useApp } from "../components/context";
import { Badge, Card, Empty, Loading, Note, SeriesKey } from "../components/ui";
import { logRetPct, modelLabel, NA, pct } from "../lib/format";
import { drawdown, percentileRank, rollingVol } from "../lib/stats";
import { useApi } from "../lib/useApi";

const EPISODES = [
  { name: "Dot-com bust", start: "2000-03-24", end: "2002-10-09" },
  { name: "Global financial crisis", start: "2007-10-09", end: "2009-03-09" },
  { name: "Euro-area debt stress", start: "2011-04-29", end: "2011-10-03" },
  { name: "COVID-19 crash", start: "2020-02-19", end: "2020-03-23" },
  { name: "2022 rate shock", start: "2022-01-03", end: "2022-10-12" },
];

/** Stored predictions for a task's selected configuration, development + holdout concatenated. */
function useSelectedHistory(mid: string, avail: AvailablePrediction[], task: string) {
  const pickFor = (period: string) => avail.find((a) => a.period === period && a.task === task && a.selected)
    ?? avail.find((a) => a.period === period && a.task === task && !a.reference);
  const dev = pickFor("dev");
  const ho = dev ? avail.find((a) => a.period === "holdout" && a.task === task && a.model === dev.model && a.feature_set === dev.feature_set) : undefined;
  return useApi(dev ? async () => {
    const parts = await Promise.all([dev, ho].filter(Boolean).map((a) =>
      api.predictions({ market: mid, task, model: a!.model, feature_set: a!.feature_set, period: a!.period })));
    return { pick: dev, rows: parts.flatMap((p) => p.rows.map((r) => ({ ...r, period: p.meta.period }))) as (PredictionRow & { period: string })[] };
  } : null, [mid, task, dev?.model, dev?.feature_set, !!ho]);
}

export function RiskPage() {
  const { market } = useApp();
  const pal = usePalette();
  const mid = market!.id;
  const px = useApi(() => api.prices(mid, "1995-01-01"), [mid]);
  const avail = useApi(() => api.available(mid), [mid]);
  const fc = useApi(() => api.forecast(mid), [mid]);
  const list = avail.data ?? [];
  const vol = useSelectedHistory(mid, list, "volatility_20");
  const down = useSelectedHistory(mid, list, "downside_20");
  const regime = useSelectedHistory(mid, list, "regime_20");

  const obs = useMemo(() => {
    if (!px.data) return null;
    const rows = px.data.rows;
    return { rows, dd: drawdown(rows.map((r) => r.close)), rv: rollingVol(rows.map((r) => r.return_1d), 21) };
  }, [px.data]);

  return (
    <div className="page">
      <div className="page-head"><div>
        <h1>Market Risk Analysis</h1>
        <p className="muted">Volatility, drawdowns, downside probabilities and regimes. Histories of forecasts are the stored
          walk-forward predictions; realised measures are computed from observed closes.</p>
      </div></div>

      <RiskAlerts fc={fc.data} vol={vol.data?.rows ?? []} down={down.data?.rows ?? []} dd={obs ? obs.dd[obs.dd.length - 1] : undefined} />

      <Card title="Volatility: forecast vs realised" subtitle={vol.data ? `20-day horizon · ${modelLabel(vol.data.pick.model)}` : undefined}
        actions={<><SeriesKey kind="actual" label="Realised 21-day vol (trailing)" /><SeriesKey kind="forecast" label="Predicted next-20-day vol" /></>}>
        {vol.loading || px.loading ? <Loading /> : obs && vol.data ? (
          <EChart height={340} ariaLabel="Volatility history" option={volOption(obs.rows, obs.rv, vol.data.rows, pal)} />
        ) : <Empty>Volatility predictions are not available yet.</Empty>}
        <p className="muted small">Forecasts are plotted at the date they were made (trailing realised vol at that date is what was known).</p>
      </Card>

      <div className="grid-2">
        <Card title="Drawdown from running peak" subtitle="Observed closes (price index, dividends excluded)">
          {obs ? <EChart height={300} ariaLabel="Drawdown" option={ddOption(obs.rows, obs.dd, pal)} /> : <Loading />}
        </Card>
        <Card title="Downside risk: P(20-day loss > 5%)" subtitle={down.data ? modelLabel(down.data.pick.model) : undefined}>
          {down.loading ? <Loading /> : down.data ? (
            <EChart height={300} ariaLabel="Downside probability history" option={probOption(down.data.rows, pal)} />
          ) : <Empty>Downside predictions not available yet.</Empty>}
        </Card>
      </div>

      <Card title="Market regime probabilities (next 20 sessions)"
        subtitle="High-vol if realised vol > 25%; otherwise bull if return > +2%, bear if < −2%, else sideways">
        {regime.loading ? <Loading /> : regime.data ? (
          <EChart height={300} ariaLabel="Regime probabilities" option={regimeOption(regime.data.rows, pal)} />
        ) : <Empty>Regime predictions not available yet.</Empty>}
      </Card>

      <Card title="Historical stress periods" subtitle="Computed from observed closes within each window">
        {obs ? <Episodes rows={obs.rows} /> : <Loading />}
      </Card>
      <Note>&quot;Risk score&quot; history is not shown: the backend defines no composite risk score. Combining the
        components above into one number would require a validated definition.</Note>
    </div>
  );
}

function RiskAlerts({ fc, vol, down, dd }: { fc: Forecast | null; vol: PredictionRow[]; down: PredictionRow[]; dd?: number }) {
  if (!fc) return <Card title="Risk alerts"><Empty>No current forecast, so no alerts. Run <code>marketpred predict</code>.</Empty></Card>;
  const h20 = fc.horizons_trading_days["20"];
  const pv = h20?.predicted_volatility_annualized ?? null;
  const pd = h20?.prob_loss_beyond_threshold ?? null;
  const volRank = pv !== null ? percentileRank(vol.map((r) => r.predicted_vol ?? NaN), pv) : NaN;
  const downRank = pd !== null ? percentileRank(down.map((r) => r.probability ?? NaN), pd) : NaN;
  const alerts: { tone: "bad" | "warn" | "good"; text: string }[] = [];
  if (Number.isFinite(volRank)) {
    alerts.push({ tone: volRank > 0.9 ? "bad" : volRank > 0.75 ? "warn" : "good",
      text: `Predicted 20-day volatility ${pct(pv)} is at the ${Math.round(volRank * 100)}th percentile of all historical forecasts.` });
  }
  if (Number.isFinite(downRank)) {
    alerts.push({ tone: downRank > 0.9 ? "bad" : downRank > 0.75 ? "warn" : "good",
      text: `P(20-day loss > 5%) = ${pct(pd)}, ${Math.round(downRank * 100)}th percentile of historical forecasts.` });
  }
  if (dd !== undefined) {
    alerts.push({ tone: dd < -0.2 ? "bad" : dd < -0.1 ? "warn" : "good", text: `Current drawdown from the running peak: ${pct(dd)}.` });
  }
  return (
    <Card title="Risk alerts" subtitle={`Based on the forecast made at the close of ${fc.data_as_of_session}`} actions={<Badge tone="forecast">Forecast</Badge>}>
      <ul className="alerts">{alerts.map((a) => <li key={a.text} className={a.tone}>{a.text}</li>)}</ul>
      <p className="muted small">Thresholds: warn above the 75th, alert above the 90th percentile of the model&apos;s own historical forecasts;
        drawdown warn at −10%, alert at −20%. These are descriptive flags, not trading signals.</p>
    </Card>
  );
}

function Episodes({ rows }: { rows: PriceRow[] }) {
  const out = EPISODES.map((e) => {
    const w = rows.filter((r) => r.date >= e.start && r.date <= e.end);
    if (w.length < 5) return { ...e, ok: false as const };
    const closes = w.map((r) => r.close);
    const rets = w.map((r) => r.return_1d).filter((x): x is number => x !== null);
    const m = rets.reduce((a, b) => a + b, 0) / rets.length;
    const v = Math.sqrt(rets.reduce((a, b) => a + (b - m) ** 2, 0) / (rets.length - 1) * 252);
    return { ...e, ok: true as const, ret: Math.log(closes[closes.length - 1] / closes[0]), mdd: Math.min(...drawdown(closes)), vol: v, n: w.length };
  });
  return (
    <div className="table-scroll"><table>
      <thead><tr><th>Episode</th><th>Window</th><th className="num">Sessions</th><th className="num">Return</th>
        <th className="num">Max drawdown</th><th className="num">Realised vol</th></tr></thead>
      <tbody>{out.map((e) => (
        <tr key={e.name}><td>{e.name}</td><td>{e.start} → {e.end}</td>
          {e.ok ? <><td className="num">{e.n}</td><td className="num">{logRetPct(e.ret)}</td><td className="num">{pct(e.mdd)}</td><td className="num">{pct(e.vol)}</td></>
            : <td colSpan={4} className="muted">{NA}: outside this market&apos;s data coverage</td>}
        </tr>))}</tbody>
    </table></div>
  );
}

function volOption(rows: PriceRow[], rv: (number | null)[], preds: PredictionRow[], p: Palette): EChartsOption {
  const dates = rows.map((r) => r.date);
  const pm = new Map(preds.map((r) => [r.prediction_date, r.predicted_vol ?? null]));
  const ax = axisStyle(p);
  return {
    ...baseOption(p), grid: { left: 56, right: 16, top: 30, bottom: 48 },
    xAxis: { type: "category", data: dates, ...ax },
    yAxis: { type: "value", ...ax, axisLabel: { ...ax.axisLabel, formatter: (v: number) => `${Math.round(v * 100)}%` } },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 18, bottom: 6 }],
    series: [
      { name: "Realised 21d vol", type: "line", data: rv, showSymbol: false, lineStyle: { color: p.actual, width: 1.2 }, itemStyle: { color: p.actual } },
      { name: "Predicted 20d vol", type: "line", data: dates.map((d) => pm.get(d) ?? null), showSymbol: false, connectNulls: false,
        lineStyle: { color: p.forecast, width: 1.4, type: "dashed" }, itemStyle: { color: p.forecast } },
    ],
  };
}

function ddOption(rows: PriceRow[], dd: number[], p: Palette): EChartsOption {
  const ax = axisStyle(p);
  return {
    ...baseOption(p), grid: { left: 56, right: 16, top: 24, bottom: 48 },
    xAxis: { type: "category", data: rows.map((r) => r.date), ...ax },
    yAxis: { type: "value", max: 0, ...ax, axisLabel: { ...ax.axisLabel, formatter: (v: number) => `${Math.round(v * 100)}%` } },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 18, bottom: 6 }],
    series: [{ name: "Drawdown", type: "line", data: dd, showSymbol: false, lineStyle: { color: p.down, width: 1 },
      areaStyle: { color: p.down, opacity: 0.18 }, itemStyle: { color: p.down } }],
  };
}

function probOption(rows: PredictionRow[], p: Palette): EChartsOption {
  const ax = axisStyle(p);
  return {
    ...baseOption(p), grid: { left: 56, right: 16, top: 30, bottom: 48 },
    xAxis: { type: "category", data: rows.map((r) => r.prediction_date), ...ax },
    yAxis: { type: "value", min: 0, max: 1, ...ax },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 18, bottom: 6 }],
    series: [
      { name: "Predicted probability", type: "line", data: rows.map((r) => r.probability ?? null), showSymbol: false,
        lineStyle: { color: p.forecast, width: 1.2 }, itemStyle: { color: p.forecast } },
      { name: "Loss occurred (1)", type: "scatter", symbolSize: 3, data: rows.map((r) => (r.actual_event ? 1 : null)),
        itemStyle: { color: p.actual, opacity: 0.5 } },
    ],
  };
}

function regimeOption(rows: PredictionRow[], p: Palette): EChartsOption {
  const ax = axisStyle(p);
  const regs = [["prob_bull", "Bull", p.up], ["prob_sideways", "Sideways", p.baseline], ["prob_bear", "Bear", p.down],
    ["prob_high_vol", "High vol", p.forecast]] as const;
  return {
    ...baseOption(p), grid: { left: 56, right: 16, top: 36, bottom: 48 },
    xAxis: { type: "category", data: rows.map((r) => r.prediction_date), ...ax },
    yAxis: { type: "value", min: 0, max: 1, ...ax },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 18, bottom: 6 }],
    series: regs.map(([k, name, color]) => ({
      name, type: "line" as const, stack: "p", data: rows.map((r) => (r[k] as number | undefined) ?? null), showSymbol: false,
      lineStyle: { width: 0 }, areaStyle: { color, opacity: 0.55 }, itemStyle: { color },
    })),
  };
}
