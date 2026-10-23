import type { EChartsOption } from "echarts";
import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { AvailablePrediction, EvaluationResponse, PredictionRow, PriceRow } from "../api/types";
import { Column, DataTable } from "../components/DataTable";
import { axisStyle, baseOption, EChart, Palette, usePalette } from "../components/EChart";
import { useApp } from "../components/context";
import {
  Badge, Card, Empty, ErrorBox, Field, Loading, Note, Segmented, Select, SeriesKey, Stat,
} from "../components/ui";
import { QUICK_RANGES, QuickRange, resolveRange, validateRange } from "../lib/dates";
import { featureSetLabel, logRetPct, modelLabel, NA, num, pct, price, pval } from "../lib/format";
import { histogram, mean } from "../lib/stats";
import { useApi } from "../lib/useApi";

type Metric = "price" | "return" | "direction" | "volatility" | "downside";

const METRICS: { value: Metric; label: string; family: string; help: string }[] = [
  { value: "price", label: "Closing price", family: "return",
    help: "Actual close at the target date vs. the price implied by the return forecast" },
  { value: "return", label: "Return", family: "return", help: "Actual vs. predicted log return over the horizon" },
  { value: "direction", label: "Direction", family: "direction", help: "Actual up/down vs. predicted probability of a rise" },
  { value: "volatility", label: "Volatility", family: "volatility", help: "Realised vs. predicted annualised volatility" },
  { value: "downside", label: "Downside risk", family: "downside", help: "Loss beyond the threshold vs. its predicted probability" },
];

const GROUP = "backtest-sync";

interface ErrorSpec { key: string; label: string; fmt: (v: unknown) => string }

function errorSpec(metric: Metric): ErrorSpec {
  switch (metric) {
    case "price": return { key: "price_error", label: "Price error (actual − predicted)", fmt: (v) => price(v) };
    case "return": return { key: "return_error", label: "Return error (actual − predicted, log)", fmt: (v) => num(v, 4) };
    case "volatility": return { key: "vol_error", label: "Vol error (actual − predicted)", fmt: (v) => pct(v, 1) };
    default: return { key: "prob_error", label: "Probability error (outcome − P)", fmt: (v) => num(v, 3) };
  }
}

export function BacktestPage() {
  const { market } = useApp();
  const pal = usePalette();
  const mid = market!.id;

  const avail = useApi(() => api.available(mid), [mid]);
  const [metric, setMetric] = useState<Metric>("price");
  const [period, setPeriod] = useState<"dev" | "holdout">("dev");
  const [horizon, setHorizon] = useState<number>(5);
  const [model, setModel] = useState<string>("");
  const [fs, setFs] = useState<string>("price");
  const [range, setRange] = useState<QuickRange>("1y");
  const [custom, setCustom] = useState<{ start: string; end: string }>({ start: "", end: "" });
  const [showErrors, setShowErrors] = useState(true);

  const family = METRICS.find((m) => m.value === metric)!.family;
  const list: AvailablePrediction[] = avail.data ?? [];
  const periods = Array.from(new Set(list.map((a) => a.period)));
  const famRows = list.filter((a) => a.period === period && a.task.replace(/_\d+$/, "") === family);
  const horizons = Array.from(new Set(famRows.map((a) => a.horizon))).sort((a, b) => a - b);
  const hRows = famRows.filter((a) => a.horizon === horizon);
  const models = Array.from(new Set(hRows.map((a) => a.model)));
  const fsets = hRows.filter((a) => a.model === model).map((a) => a.feature_set);
  const task = `${family}_${horizon}`;

  // Keep selections valid as options change; prefer the pre-registered selected configuration.
  useEffect(() => {
    if (horizons.length && !horizons.includes(horizon)) setHorizon(horizons.includes(5) ? 5 : horizons[0]);
  }, [horizons.join(","), horizon]);
  useEffect(() => {
    if (!models.length) return;
    if (!models.includes(model)) {
      const sel = hRows.find((a) => a.selected) ?? hRows.find((a) => !a.reference) ?? hRows[0];
      setModel(sel.model);
      setFs(sel.feature_set);
    }
  }, [models.join(","), model]);
  useEffect(() => {
    if (fsets.length && !fsets.includes(fs)) setFs(fsets.includes("price") ? "price" : fsets[0]);
  }, [fsets.join(","), fs]);
  useEffect(() => {
    if (periods.length && !periods.includes(period)) setPeriod(periods[0] as "dev" | "holdout");
  }, [periods.join(",")]);

  const ready = !!model && fsets.includes(fs) && models.includes(model);
  const preds = useApi(ready ? () => api.predictions({ market: mid, task, model, feature_set: fs, period }) : null,
    [mid, task, model, fs, period, ready]);
  const allRows = preds.data?.rows ?? [];
  const dates = useMemo(() => allRows.map((r) => r.prediction_date), [allRows]);

  const win = useMemo(() => {
    if (range === "custom") return custom.start && custom.end ? custom : null;
    return resolveRange(range, dates);
  }, [range, custom, dates]);
  useEffect(() => {  // seed the custom inputs from the current quick range
    if (range !== "custom" && win) setCustom(win);
  }, [win?.start, win?.end]);

  const check = win ? validateRange(win.start, win.end, dates) : null;
  const rows = useMemo(() => (win ? allRows.filter((r) => r.prediction_date >= win.start
    && r.prediction_date <= win.end) : []), [allRows, win]);

  const lastTarget = rows.length ? rows[rows.length - 1].target_date : undefined;
  const px = useApi(rows.length ? () => api.prices(mid, rows[0].prediction_date, lastTarget) : null,
    [mid, rows[0]?.prediction_date, lastTarget]);
  const evalOk = !!(win && check?.ok && check.count >= 20);
  const ev = useApi(evalOk ? () => api.evaluation({ market: mid, task, model, feature_set: fs, period,
    start: win!.start, end: win!.end }) : null, [mid, task, model, fs, period, win?.start, win?.end, evalOk]);

  // Same-model results across horizons of this family (or the selected config where the model is absent).
  const byHorizon = useApi(evalOk && horizons.length ? () => Promise.all(horizons.map(async (h) => {
    const rowsH = famRows.filter((a) => a.horizon === h);
    const pick = rowsH.find((a) => a.model === model && a.feature_set === fs) ?? rowsH.find((a) => a.selected) ?? rowsH[0];
    try {
      const e = await api.evaluation({ market: mid, task: `${family}_${h}`, model: pick.model,
        feature_set: pick.feature_set, period, start: win!.start, end: win!.end });
      return { h, pick, e, err: null as string | null };
    } catch (err) {
      return { h, pick, e: null, err: String((err as Error).message) };
    }
  })) : null, [mid, family, model, fs, period, win?.start, win?.end, evalOk, horizons.join(",")]);

  if (avail.loading) return <Loading label="Loading prediction inventory…" />;
  if (avail.error) return <ErrorBox error={avail.error.message} />;
  if (!list.length) {
    return (
      <Card title="Historical Backtesting & Prediction Comparison">
        <Empty>No stored walk-forward predictions for {market!.name} yet. Run <code>marketpred evaluate</code>
          {market!.id !== "SPX" && <> with <code>--config configs/{market!.id.toLowerCase()}.yaml</code></>}.</Empty>
      </Card>
    );
  }

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Historical Backtesting &amp; Prediction Comparison</h1>
          <p className="muted">Compare what {market!.name} actually did with what the model predicted for the same
            dates, using only forecasts produced without access to the future.</p>
        </div>
        <div className="row gap">
          <Badge tone="good" title={preds.data?.meta.method}>Walk-forward out-of-sample</Badge>
          <Badge tone="neutral">No retrospective fits</Badge>
        </div>
      </div>

      <Card className="filters">
        <div className="filter-grid">
          <Field label="Instrument"><input value={`${market!.name} (${market!.ticker})`} readOnly aria-readonly /></Field>
          <Field label="Compare" help={METRICS.find((m) => m.value === metric)?.help}>
            <Select value={metric} onChange={setMetric} options={METRICS.map((m) => ({ value: m.value, label: m.label,
              disabled: !list.some((a) => a.period === period && a.task.startsWith(m.family)) }))} />
          </Field>
          <Field label="Horizon (trading days)" help="Only horizons the models were trained for are offered">
            <Select value={String(horizon)} onChange={(v) => setHorizon(Number(v))}
              options={horizons.map((h) => ({ value: String(h), label: `${h} day${h > 1 ? "s" : ""}` }))} />
          </Field>
          <Field label="Model">
            <Select value={model} onChange={setModel} options={models.map((m) => ({ value: m,
              label: `${modelLabel(m)}${hRows.some((a) => a.model === m && a.selected) ? " ★ selected" : ""}` }))} />
          </Field>
          <Field label="Feature configuration" help="Category sets evaluated by the backend">
            <Select value={fs} onChange={setFs} options={fsets.map((f) => ({ value: f, label: featureSetLabel(f) }))} />
          </Field>
          <Field label="Evaluation period" help="Development = used for model selection; Holdout = untouched final test">
            <Select value={period} onChange={(v) => setPeriod(v)} options={[
              { value: "dev", label: "Development (walk-forward)", disabled: !periods.includes("dev") },
              { value: "holdout", label: "Final holdout", disabled: !periods.includes("holdout") },
            ]} />
          </Field>
        </div>
        <div className="range-row">
          <Segmented value={range} onChange={setRange} options={QUICK_RANGES.map((r) => ({ value: r.id, label: r.label }))} />
          <div className="row gap">
            <Field label="Start">
              <input type="date" value={custom.start} min={dates[0]} max={dates[dates.length - 1]}
                onChange={(e) => { setRange("custom"); setCustom({ ...custom, start: e.target.value }); }} />
            </Field>
            <Field label="End">
              <input type="date" value={custom.end} min={dates[0]} max={dates[dates.length - 1]}
                onChange={(e) => { setRange("custom"); setCustom({ ...custom, end: e.target.value }); }} />
            </Field>
          </div>
        </div>
        {dates.length > 0 && (
          <p className="muted small">Predictions available for this selection: {dates[0]} → {dates[dates.length - 1]}
            {" "}({dates.length.toLocaleString()} trading days). Ranges end at the last date with an observed outcome.</p>
        )}
        {check?.message && <Note tone={check.ok ? "info" : "warn"}>{check.message}</Note>}
        {period === "dev" && (
          <Note>Each development prediction is out-of-sample, but the <em>choice</em> of the best model was made on
            this period, so the best-looking configuration is optimistically biased here. The final holdout is the
            unbiased check{periods.includes("holdout") ? "" : " (not evaluated yet)"}.</Note>
        )}
      </Card>

      {preds.loading && <Loading label="Loading stored predictions…" />}
      {preds.error && <ErrorBox error={preds.error.message} />}

      {preds.data && rows.length > 0 && (
        <>
          <SummaryPanel metric={metric} rows={rows} ev={ev.data} evLoading={ev.loading} evError={ev.error?.message}
            insufficient={!evalOk} />
          <div className="chart-pair">
            <Card title="Chart 1 · Actual market data" subtitle="Observed end-of-day prices and volume (delayed, not live)"
              actions={<SeriesKey kind="actual" label="Actual" />}>
              {px.loading && <Loading />}
              {px.error && <ErrorBox error={px.error.message} />}
              {px.data && <EChart group={GROUP} height={440} ariaLabel="Actual market prices"
                option={actualOption(px.data.rows, px.data.open_reliable_from, pal)} />}
              {px.data?.open_reliable_from && rows[0].prediction_date < px.data.open_reliable_from && (
                <p className="muted small">Open prices before {px.data.open_reliable_from} are unreliable in the source data, so
                  closes are drawn as a line instead of candlesticks.</p>
              )}
            </Card>
            <Card title="Chart 2 · Model predictions" subtitle={`${modelLabel(model)} · ${featureSetLabel(fs)} · ${horizon}-day horizon, plotted at each target date`}
              actions={<><SeriesKey kind="forecast" label="Predicted" /><SeriesKey kind="baseline" label="Actual (reference)" /></>}>
              {px.data ? <EChart group={GROUP} height={440} ariaLabel="Model predictions versus actual outcomes"
                option={predictionOption(metric, rows, px.data.rows, pal, preds.data.meta.horizon)} /> : <Loading />}
              <p className="muted small">{chart2Caption(metric, preds.data.meta.price_conversion)}</p>
            </Card>
          </div>

          <HorizonTable data={byHorizon.data} loading={byHorizon.loading} metric={metric} />

          <Card title="Error and deviation analysis" actions={
            <button type="button" className="btn ghost" onClick={() => setShowErrors(!showErrors)}>
              {showErrors ? "Hide" : "Show"}</button>}>
            {showErrors && <ErrorAnalysis metric={metric} rows={rows} ev={ev.data} pal={pal} />}
          </Card>

          <Card title="Prediction-by-prediction comparison"
            subtitle="One row per evaluated prediction. Training end = last date of the data used to fit the model that made the prediction.">
            <DataTable rows={rows as unknown as Record<string, unknown>[]} columns={tableColumns(metric) as Column<Record<string, unknown>>[]}
              csvName={`${mid}_${task}_${model}_${fs}_${period}_${win?.start}_${win?.end}.csv`}
              filter={(r, q) => String(r.prediction_date).includes(q) || String(r.target_date).includes(q)} />
          </Card>

          <Card title="How these historical predictions were produced">
            <ul className="method">
              <li><strong>Walk-forward, out-of-sample:</strong> {preds.data.meta.method}. {preds.data.meta.n_folds} refits in this period.</li>
              <li><strong>Point-in-time inputs:</strong> every feature used only values published before that day's market
                close (e.g. VIX and Treasury yields from the previous session; monthly macro after release lags).</li>
              <li><strong>Target:</strong> {preds.data.meta.target} over {preds.data.meta.horizon} trading session(s) from the
                prediction date&apos;s close, counted on the exchange calendar (weekends and holidays skipped).</li>
              <li><strong>Outcomes:</strong> only predictions whose target date has passed are stored and shown; the latest live
                forecast appears on the Market Overview page.</li>
              {preds.data.meta.price_conversion && <li><strong>Price conversion:</strong> {preds.data.meta.price_conversion}.</li>}
              <li><strong>Probabilities:</strong> {preds.data.meta.calibrated ? "Platt-calibrated inside each training window." :
                "not post-calibrated; check the calibration error (ECE) above."}</li>
            </ul>
          </Card>
        </>
      )}
      {preds.data && !rows.length && check && !check.ok && <Empty>{check.message}</Empty>}
    </div>
  );
}

function chart2Caption(metric: Metric, conversion: string | null): string {
  switch (metric) {
    case "price": return `Predicted close = ${conversion ?? "derived from the return forecast"}. The return model gives a point forecast only; no predictive interval is stored, so none is drawn.`;
    case "return": return "Bars: realised log return over the horizon ending at the target date. Line: the model's forecast made h sessions earlier.";
    case "direction": return "Line: predicted probability of a positive return. Markers: what actually happened (1 = up, 0 = down). Predicted direction = up when P > 0.5.";
    case "volatility": return "Annualised realised volatility over the horizon ending at the target date vs. the forecast made at its start.";
    default: return "Line: predicted probability of a loss beyond the threshold. Markers: whether the loss occurred.";
  }
}

// ---- charts ------------------------------------------------------------------------------------------
function zoom(axes: number[] = [0]): EChartsOption["dataZoom"] {
  return [{ type: "inside", xAxisIndex: axes }, { type: "slider", xAxisIndex: axes, height: 18, bottom: 6 }];
}

function actualOption(prices: PriceRow[], openFrom: string | null, p: Palette): EChartsOption {
  const dates = prices.map((r) => r.date);
  const candles = !openFrom || (dates[0] ?? "") >= openFrom;
  const ax = axisStyle(p);
  return {
    ...baseOption(p),
    axisPointer: { link: [{ xAxisIndex: "all" }] },
    grid: [{ left: 64, right: 16, top: 30, height: "58%" }, { left: 64, right: 16, top: "76%", height: "12%" }],
    xAxis: [
      { type: "category", data: dates, boundaryGap: true, ...ax, axisLabel: { show: false } },
      { type: "category", data: dates, gridIndex: 1, ...ax },
    ],
    yAxis: [
      { type: "value", scale: true, ...ax, name: "Price" },
      { type: "value", gridIndex: 1, ...ax, axisLabel: { show: false }, splitLine: { show: false } },
    ],
    dataZoom: zoom([0, 1]),
    series: [
      candles
        ? { name: "Actual OHLC", type: "candlestick", data: prices.map((r) => [r.open, r.close, r.low, r.high]),
            itemStyle: { color: p.up, color0: p.down, borderColor: p.up, borderColor0: p.down } }
        : { name: "Actual close", type: "line", data: prices.map((r) => r.close), showSymbol: false,
            lineStyle: { color: p.actual, width: 1.6 }, itemStyle: { color: p.actual } },
      { name: "Volume", type: "bar", xAxisIndex: 1, yAxisIndex: 1, data: prices.map((r) => r.volume),
        itemStyle: { color: p.baseline, opacity: 0.5 } },
    ],
  };
}

function predictionOption(metric: Metric, rows: PredictionRow[], prices: PriceRow[], p: Palette, h: number): EChartsOption {
  const dates = prices.map((r) => r.date);
  const at = new Map<string, PredictionRow>(rows.map((r) => [r.target_date, r]));
  const pick = (f: (r: PredictionRow) => unknown) => dates.map((d) => {
    const r = at.get(d);
    const v = r ? f(r) : null;
    return typeof v === "number" && Number.isFinite(v) ? v : null;
  });
  const ax = axisStyle(p);
  const base: EChartsOption = {
    ...baseOption(p),
    grid: { left: 64, right: 16, top: 30, bottom: 48 },
    xAxis: { type: "category", data: dates, boundaryGap: true, ...ax },
    dataZoom: zoom(),
  };
  const fc = { color: p.forecast };
  if (metric === "price") {
    return { ...base, yAxis: { type: "value", scale: true, name: "Price", ...ax }, series: [
      { name: "Actual close", type: "line", data: prices.map((r) => r.close), showSymbol: false,
        lineStyle: { color: p.baseline, width: 1.2 }, itemStyle: { color: p.baseline } },
      { name: `Predicted close (${h}d ahead)`, type: "line", data: pick((r) => r.predicted_price), showSymbol: rows.length < 120,
        symbolSize: 4, connectNulls: false, lineStyle: { ...fc, type: "dashed", width: 1.6 }, itemStyle: fc },
    ] };
  }
  if (metric === "return") {
    return { ...base, yAxis: { type: "value", name: "Log return", ...ax }, series: [
      { name: `Actual ${h}d return`, type: "bar", data: pick((r) => r.actual_return), itemStyle: { color: p.baseline, opacity: 0.55 } },
      { name: "Predicted return", type: "line", data: pick((r) => r.predicted_return), showSymbol: false,
        lineStyle: { ...fc, type: "dashed", width: 1.8 }, itemStyle: fc },
    ] };
  }
  if (metric === "volatility") {
    return { ...base, yAxis: { type: "value", name: "Annualised vol", ...ax, axisLabel: { ...ax.axisLabel, formatter: (v: number) => `${Math.round(v * 100)}%` } }, series: [
      { name: `Realised ${h}d vol`, type: "line", data: pick((r) => r.actual_vol), showSymbol: false,
        lineStyle: { color: p.baseline, width: 1.2 }, itemStyle: { color: p.baseline } },
      { name: "Predicted vol", type: "line", data: pick((r) => r.predicted_vol), showSymbol: false,
        lineStyle: { ...fc, type: "dashed", width: 1.8 }, itemStyle: fc },
    ] };
  }
  // binary: direction or downside
  const label = metric === "direction" ? "P(up)" : "P(loss beyond threshold)";
  return { ...base, yAxis: { type: "value", min: 0, max: 1, name: "Probability / outcome", ...ax }, series: [
    { name: `Actual outcome (1 = ${metric === "direction" ? "up" : "loss"})`, type: "scatter", symbolSize: 5,
      data: pick((r) => r.actual_event), itemStyle: { color: p.baseline, opacity: 0.6 } },
    { name: `Predicted ${label}`, type: "line", data: pick((r) => r.probability), showSymbol: false,
      lineStyle: { ...fc, width: 1.8 }, itemStyle: fc,
      markLine: metric === "direction" ? { silent: true, symbol: "none", lineStyle: { color: p.muted, type: "dotted" },
        data: [{ yAxis: 0.5, name: "0.5" }] } : undefined },
  ] };
}

// ---- summary -----------------------------------------------------------------------------------------
function SummaryPanel({ metric, rows, ev, evLoading, evError, insufficient }: {
  metric: Metric; rows: PredictionRow[]; ev: EvaluationResponse | null; evLoading: boolean; evError?: string;
  insufficient: boolean;
}) {
  const m = ev?.metrics ?? {};
  const b = ev?.baseline;
  const na = (why: string) => <span title={why}>{NA}</span>;
  const dirAcc = metric === "direction" ? pct(m.accuracy) : (metric === "price" || metric === "return") ? pct(m.hit_rate) : null;
  const volAbs = metric === "volatility" ? rows.map((r) => Math.abs((r.actual_vol ?? NaN) - (r.predicted_vol ?? NaN))) : [];
  return (
    <Card title="Prediction accuracy summary" subtitle={ev ? `${ev.start} → ${ev.end} · ${ev.n} evaluated predictions` : undefined}>
      {evLoading && <Loading label="Computing metrics…" />}
      {evError && <ErrorBox error={evError} />}
      {insufficient && <Note tone="warn">Fewer than 20 predictions in the window, so summary metrics are not computed
        (they would be too noisy to interpret). Widen the date range.</Note>}
      {ev && (
        <div className="stat-grid">
          <Stat label="Predictions evaluated" value={ev.n.toLocaleString()} />
          <Stat label="Directional accuracy" value={dirAcc ?? na("Not a directional target")}
            hint={metric === "direction" ? `Base rate of up moves: ${pct(m.base_rate)}` : (metric === "price" || metric === "return") ? "Share of correct signs" : undefined} />
          {metric === "price" && <>
            <Stat label="MAE (price)" value={price(ev.price_metrics?.mae)} />
            <Stat label="RMSE (price)" value={price(ev.price_metrics?.rmse)} />
            <Stat label="MAPE" value={pct((ev.price_metrics?.mape_pct ?? NaN) / 100, 2)}
              hint="Meaningful for price levels; most of the level is known at prediction time" />
          </>}
          {(metric === "price" || metric === "return") && <>
            <Stat label="Return MAE" value={num(m.mae, 4)} hint="Log-return units" />
            <Stat label="Return RMSE" value={num(m.rmse, 4)} />
            <Stat label="Out-of-sample R²" value={num(m.r2_oos, 4)} tone={(m.r2_oos ?? 0) > 0 ? "good" : "bad"}
              hint="vs. the historical mean of each training window (Campbell–Thompson)" />
          </>}
          {metric === "volatility" && <>
            <Stat label="MAE (vol)" value={pct(mean(volAbs), 2)} hint="Annualised volatility points, computed from the rows shown" />
            <Stat label="RMSE (vol)" value={pct(Math.sqrt(mean(volAbs.map((x) => x * x))), 2)} />
            <Stat label="MSE of log vol" value={num(m.mse_log, 4)} />
            <Stat label="QLIKE" value={num(m.qlike, 4)} hint="Robust volatility loss (lower is better)" />
            <Stat label="Mincer–Zarnowitz R²" value={num(m.mz_r2, 3)} />
          </>}
          {(metric === "direction" || metric === "downside") && <>
            <Stat label="ROC-AUC" value={num(m.auc, 3)} tone={(m.auc ?? 0.5) > 0.55 ? "good" : "muted"} hint="0.5 = no discrimination" />
            <Stat label="Precision" value={pct(m.precision)} />
            <Stat label="Recall" value={pct(m.recall)} />
            <Stat label="F1" value={num(m.f1, 3)} />
            <Stat label="Brier score" value={num(m.brier, 4)} />
            <Stat label="Calibration error (ECE)" value={num(m.ece, 3)} />
          </>}
        </div>
      )}
      {ev && (
        <div className="baseline-box">
          {b ? (
            <>
              <strong>Versus baseline ({modelLabel(b.model)}):</strong>{" "}
              loss {num(b.comparison.loss_ref, 4)} → {num(b.comparison.loss_model, 4)}{" "}
              ({b.comparison.rel_improvement >= 0 ? "−" : "+"}{Math.abs(b.comparison.rel_improvement * 100).toFixed(2)}% loss),
              95% CI of improvement [{num(b.comparison.ci_low, 4)}, {num(b.comparison.ci_high, 4)}],
              Diebold–Mariano p = {pval(b.comparison.p_value)}.{" "}
              {b.comparison.p_value < 0.05 && b.comparison.loss_improvement > 0
                ? <Badge tone="good">significantly better than baseline</Badge>
                : b.comparison.p_value < 0.05 ? <Badge tone="bad">significantly worse than baseline</Badge>
                  : <Badge tone="neutral">not distinguishable from baseline</Badge>}
            </>
          ) : <span className="muted">Baseline comparison: {NA} (the selected model is the baseline, or baseline predictions are not stored).</span>}
        </div>
      )}
    </Card>
  );
}

function HorizonTable({ data, loading, metric }: {
  data: { h: number; pick: AvailablePrediction; e: EvaluationResponse | null; err: string | null }[] | null;
  loading: boolean; metric: Metric;
}) {
  if (loading) return <Card title="Performance by forecast horizon"><Loading /></Card>;
  if (!data?.length) return null;
  const key = metric === "volatility" ? ["mse_log", "qlike", "mz_r2"] : (metric === "direction" || metric === "downside")
    ? ["accuracy", "auc", "brier", "log_loss"] : ["hit_rate", "rmse", "r2_oos"];
  return (
    <Card title="Performance by forecast horizon" subtitle="Same window; same model where trained for that horizon, otherwise the selected one">
      <div className="table-scroll"><table>
        <thead><tr><th>Horizon</th><th>Model</th><th>Features</th><th className="num">n</th>
          {key.map((k) => <th key={k} className="num">{k}</th>)}<th className="num">vs baseline p</th></tr></thead>
        <tbody>{data.map(({ h, pick, e, err }) => (
          <tr key={h}><td>{h}d</td><td>{modelLabel(pick.model)}</td><td>{featureSetLabel(pick.feature_set)}</td>
            {e ? <>
              <td className="num">{e.n}</td>
              {key.map((k) => <td key={k} className="num">{num(e.metrics[k], 4)}</td>)}
              <td className="num">{e.baseline ? pval(e.baseline.comparison.p_value) : NA}</td>
            </> : <td colSpan={key.length + 2} className="muted">{err}</td>}
          </tr>))}
        </tbody>
      </table></div>
    </Card>
  );
}

// ---- error analysis ----------------------------------------------------------------------------------
function ErrorAnalysis({ metric, rows, ev, pal }: { metric: Metric; rows: PredictionRow[]; ev: EvaluationResponse | null; pal: Palette }) {
  const spec = errorSpec(metric);
  const errs = rows.map((r) => (typeof r[spec.key] === "number" ? (r[spec.key] as number) : NaN));
  const valid = rows.map((r, i) => ({ r, e: errs[i] })).filter((x) => Number.isFinite(x.e));
  const ax = axisStyle(pal);
  const xy = metric === "price" ? (r: PredictionRow) => [r.predicted_price, r.actual_close_target]
    : metric === "return" ? (r: PredictionRow) => [r.predicted_return, r.actual_return]
      : metric === "volatility" ? (r: PredictionRow) => [r.predicted_vol, r.actual_vol]
        : (r: PredictionRow) => [r.probability, r.actual_event];
  const hist = histogram(valid.map((x) => x.e), 30);
  const worst = [...valid].sort((a, b) => Math.abs(b.e) - Math.abs(a.e)).slice(0, 10);
  const regimeKey = metric === "volatility" ? "mse_log" : (metric === "direction" || metric === "downside") ? "auc" : "rmse";
  return (
    <div className="grid-2">
      <div>
        <h3 className="h3">{spec.label} over time</h3>
        <EChart height={260} ariaLabel="Prediction error over time" option={{
          ...baseOption(pal), grid: { left: 60, right: 12, top: 24, bottom: 40 },
          xAxis: { type: "category", data: valid.map((x) => x.r.target_date), ...ax },
          yAxis: { type: "value", ...ax }, dataZoom: [{ type: "inside" }],
          series: [
            { name: "Error", type: "line", data: valid.map((x) => x.e), showSymbol: false, lineStyle: { color: pal.forecast, width: 1 } },
            { name: "|Error|", type: "bar", data: valid.map((x) => Math.abs(x.e)), itemStyle: { color: pal.baseline, opacity: 0.35 } },
          ],
        }} />
      </div>
      <div>
        <h3 className="h3">Actual vs. predicted</h3>
        <EChart height={260} ariaLabel="Actual versus predicted scatter" option={{
          ...baseOption(pal), tooltip: { trigger: "item" }, grid: { left: 60, right: 12, top: 24, bottom: 40 },
          xAxis: { type: "value", name: "Predicted", scale: true, ...ax }, yAxis: { type: "value", name: "Actual", scale: true, ...ax },
          series: [{ type: "scatter", symbolSize: 4, data: rows.map(xy), itemStyle: { color: pal.actual, opacity: 0.5 } }],
        }} />
      </div>
      <div>
        <h3 className="h3">Error distribution</h3>
        <EChart height={240} ariaLabel="Error distribution" option={{
          ...baseOption(pal), tooltip: { trigger: "axis" }, grid: { left: 50, right: 12, top: 20, bottom: 40 },
          xAxis: { type: "category", data: hist.map((b) => spec.fmt((b.x0 + b.x1) / 2)), ...ax },
          yAxis: { type: "value", ...ax },
          series: [{ type: "bar", data: hist.map((b) => b.count), itemStyle: { color: pal.forecast, opacity: 0.7 } }],
        }} />
      </div>
      <div>
        <h3 className="h3">Accuracy by volatility regime</h3>
        {ev && Object.keys(ev.by_vol_regime).length ? (
          <>
            <table><thead><tr><th>Regime</th><th className="num">n</th><th className="num">{regimeKey}</th>
              <th className="num">loss improvement vs baseline</th></tr></thead>
              <tbody>{["low_vol", "mid_vol", "high_vol"].filter((k) => ev.by_vol_regime[k]).map((k) => (
                <tr key={k}><td>{k.replace("_", " ")}</td><td className="num">{ev.by_vol_regime[k].n}</td>
                  <td className="num">{num(ev.by_vol_regime[k][regimeKey], 4)}</td>
                  <td className="num">{num(ev.by_vol_regime_improvement_vs_baseline?.[`impr_${k}`], 5)}</td></tr>))}
              </tbody></table>
            <p className="muted small">{ev.vol_regime_definition}. Regimes need ≥ 20 predictions to be shown.</p>
          </>
        ) : <Empty>Regime breakdown needs at least 20 predictions per regime.</Empty>}
        <h3 className="h3">Largest deviations</h3>
        <table><thead><tr><th>Prediction date</th><th>Target date</th><th className="num">Error</th></tr></thead>
          <tbody>{worst.map((x) => (
            <tr key={x.r.prediction_date}><td>{x.r.prediction_date}</td><td>{x.r.target_date}</td>
              <td className="num">{spec.fmt(x.e)}</td></tr>))}</tbody></table>
      </div>
    </div>
  );
}

// ---- table columns -----------------------------------------------------------------------------------
function tableColumns(metric: Metric): Column<PredictionRow>[] {
  const base: Column<PredictionRow>[] = [
    { key: "prediction_date", label: "Prediction date", title: "Close at which the forecast was made" },
    { key: "target_date", label: "Target date", title: "Session whose close resolves the forecast" },
  ];
  const dirCols: Column<PredictionRow>[] = [
    { key: "actual_direction", label: "Actual dir.", render: (r) => <DirTag d={r.actual_direction} /> },
    { key: "predicted_direction", label: "Predicted dir.", render: (r) => <DirTag d={r.predicted_direction} forecast /> },
  ];
  const tail: Column<PredictionRow>[] = [{ key: "train_end", label: "Training data end", title: "Last date in the model's training set" }];
  switch (metric) {
    case "price":
      return [...base,
        { key: "close_at_prediction", label: "Close at prediction", align: "right", render: (r) => price(r.close_at_prediction) },
        { key: "actual_close_target", label: "Actual close", align: "right", render: (r) => price(r.actual_close_target) },
        { key: "predicted_price", label: "Predicted close", align: "right", render: (r) => price(r.predicted_price) },
        { key: "actual_return", label: "Actual return", align: "right", render: (r) => logRetPct(r.actual_return) },
        { key: "predicted_return", label: "Predicted return", align: "right", render: (r) => logRetPct(r.predicted_return) },
        ...dirCols,
        { key: "price_error", label: "Error", align: "right", render: (r) => price(r.price_error) },
        { key: "price_error_pct", label: "Error %", align: "right", render: (r) => pct((r.price_error_pct ?? NaN) / 100, 2) },
        ...tail];
    case "return":
      return [...base,
        { key: "actual_return", label: "Actual return", align: "right", render: (r) => logRetPct(r.actual_return) },
        { key: "predicted_return", label: "Predicted return", align: "right", render: (r) => logRetPct(r.predicted_return) },
        ...dirCols,
        { key: "return_error", label: "Error (log)", align: "right", render: (r) => num(r.return_error, 4) },
        ...tail];
    case "volatility":
      return [...base,
        { key: "actual_vol", label: "Actual vol", align: "right", render: (r) => pct(r.actual_vol) },
        { key: "predicted_vol", label: "Predicted vol", align: "right", render: (r) => pct(r.predicted_vol) },
        { key: "actual_return", label: "Actual return", align: "right", render: (r) => logRetPct(r.actual_return) },
        { key: "vol_error", label: "Error", align: "right", render: (r) => pct(r.vol_error) },
        { key: "vol_error_pct", label: "Error %", align: "right", render: (r) => pct((r.vol_error_pct ?? NaN) / 100, 1) },
        ...tail];
    case "direction":
      return [...base,
        { key: "close_at_prediction", label: "Close at prediction", align: "right", render: (r) => price(r.close_at_prediction) },
        { key: "actual_close_target", label: "Actual close", align: "right", render: (r) => price(r.actual_close_target) },
        { key: "actual_return", label: "Actual return", align: "right", render: (r) => logRetPct(r.actual_return) },
        { key: "probability", label: "P(up)", align: "right", render: (r) => pct(r.probability) },
        ...dirCols,
        { key: "correct", label: "Correct?", value: (r) => (r.actual_direction === r.predicted_direction ? 1 : 0),
          render: (r) => (r.actual_direction === r.predicted_direction ? "✓" : "✗") },
        ...tail];
    default:
      return [...base,
        { key: "actual_return", label: "Actual return", align: "right", render: (r) => logRetPct(r.actual_return) },
        { key: "probability", label: "P(loss beyond threshold)", align: "right", render: (r) => pct(r.probability) },
        { key: "actual_event", label: "Loss occurred?", render: (r) => (r.actual_event ? "yes" : "no") },
        { key: "prob_error", label: "Outcome − P", align: "right", render: (r) => num(r.prob_error, 3) },
        ...tail];
  }
}

function DirTag({ d, forecast }: { d?: string; forecast?: boolean }) {
  if (!d) return <>{NA}</>;
  return <span className={`dir ${d} ${forecast ? "fc" : ""}`}>{d === "up" ? "▲ up" : "▼ down"}</span>;
}
