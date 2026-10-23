import type { EChartsOption } from "echarts";
import { useMemo } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { AvailablePrediction, Forecast, PriceRow } from "../api/types";
import { axisStyle, baseOption, EChart, Palette, usePalette } from "../components/EChart";
import { useApp } from "../components/context";
import { Badge, Card, Empty, ErrorBox, Loading, Note, SeriesKey, Stat } from "../components/ui";
import { logRetPct, modelLabel, NA, pct, price, taskLabel } from "../lib/format";
import { drawdown, sma } from "../lib/stats";
import { useApi } from "../lib/useApi";

export function OverviewPage() {
  const { market } = useApp();
  const pal = usePalette();
  const mid = market!.id;
  const start = useMemo(() => {
    const d = new Date(`${market!.price_coverage.end}T00:00:00Z`);
    d.setUTCFullYear(d.getUTCFullYear() - 2);
    return d.toISOString().slice(0, 10);
  }, [market]);
  const px = useApi(() => api.prices(mid, start), [mid, start]);
  const fc = useApi(() => api.forecast(mid), [mid]);
  const avail = useApi(() => api.available(mid), [mid]);

  const trend = useMemo(() => (px.data ? observedTrend(px.data.rows) : null), [px.data]);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>{market!.name}</h1>
          <p className="muted">{market!.ticker} · {market!.calendar} calendar · {market!.currency}</p>
        </div>
        <Link className="btn primary" to="/backtest">Open historical backtesting →</Link>
      </div>

      <div className="grid-2">
        <Card title="Latest observed market data" actions={<Badge tone="actual">Observed</Badge>}
          subtitle="End-of-day data from the last ingestion; delayed, not a live quote">
          {px.loading && <Loading />}
          {px.error && <ErrorBox error={px.error.message} />}
          {px.data && trend && (
            <div className="stat-grid">
              <Stat label="Last session" value={trend.last.date} hint="Close of this session is the latest data point" />
              <Stat label="Close" value={price(trend.last.close)} />
              <Stat label="1-day return" value={logRetPct(trend.last.return_1d)} tone={(trend.last.return_1d ?? 0) >= 0 ? "good" : "bad"} />
              <Stat label="Observed trend" value={trend.label} hint={trend.rule} />
              <Stat label="Drawdown from 2-year high" value={pct(trend.dd, 1)} tone={trend.dd < -0.1 ? "bad" : "muted"} />
            </div>
          )}
        </Card>
        <ForecastCard fc={fc.data} loading={fc.loading} error={fc.error?.status === 404 ? null : fc.error?.message} />
      </div>

      <Card title="Price history with latest forecast" subtitle="Two years of observed closes; the forecast band is drawn after the last session"
        actions={<><SeriesKey kind="actual" label="Observed close" /><SeriesKey kind="forecast" label="Forecast (90% band)" /></>}>
        {px.data ? <EChart height={380} ariaLabel="Price history with forecast" option={historyOption(px.data.rows, fc.data, pal)} /> : <Loading />}
        {fc.data && <p className="muted small">The band uses the latest forecast&apos;s expected return ± 1.645 × predicted volatility
          (normal approximation). Positions to the right of the last close are counted in trading sessions, not calendar days.</p>}
      </Card>

      <div className="grid-2">
        <Card title="What drives the latest forecast" subtitle="Feature categories used by each selected model">
          {fc.data ? <Drivers fc={fc.data} /> : <Empty>No forecast generated yet.</Empty>}
          <Note>The backend does not yet compute per-prediction attribution (e.g. SHAP), so no &quot;top factors&quot; are
            claimed here. Category-level evidence from out-of-sample ablations is on the <Link to="/factors">Features &amp; Factors</Link> page.</Note>
        </Card>
        <RecentPredictions mid={mid} avail={avail.data ?? []} />
      </div>
    </div>
  );
}

function observedTrend(rows: PriceRow[]) {
  const closes = rows.map((r) => r.close);
  const s50 = sma(closes, 50);
  const s200 = sma(closes, 200);
  const i = rows.length - 1;
  const last = rows[i];
  const a = s50[i];
  const b = s200[i];
  let label = NA;
  if (a !== null && b !== null) {
    label = last.close > b && a > b ? "Uptrend" : last.close < b && a < b ? "Downtrend" : "Mixed";
  }
  return { last, label, dd: drawdown(closes)[i], rule: "Uptrend: close and 50-day average above the 200-day average; Downtrend: both below; otherwise Mixed. A description of past prices, not a forecast." };
}

function ForecastCard({ fc, loading, error }: { fc: Forecast | null; loading: boolean; error?: string | null }) {
  if (loading) return <Card title="Latest forecast"><Loading /></Card>;
  if (error) return <Card title="Latest forecast"><ErrorBox error={error} /></Card>;
  if (!fc) {
    return (
      <Card title="Latest forecast" actions={<Badge tone="forecast">Forecast</Badge>}>
        <Empty>No forecast has been generated yet. Run <code>marketpred predict</code> after the evaluation completes.</Empty>
      </Card>
    );
  }
  const h = fc.horizons_trading_days;
  const keys = Object.keys(h).sort((a, b) => Number(a) - Number(b));
  return (
    <Card title="Latest forecast" actions={<Badge tone="forecast">Forecast</Badge>}
      subtitle={`Made at the close of ${fc.data_as_of_session} (${fc.prediction_timestamp_utc.slice(0, 16).replace("T", " ")} UTC)`}>
      <div className="table-scroll"><table>
        <thead><tr><th>Horizon</th><th className="num">P(up)</th><th className="num">Expected return</th>
          <th className="num">Predicted vol</th><th className="num">95% VaR</th><th className="num">P(loss beyond threshold)</th></tr></thead>
        <tbody>{keys.map((k) => (
          <tr key={k}><td>{k}d</td><td className="num">{pct(h[k].prob_positive_return)}</td>
            <td className="num">{logRetPct(h[k].expected_log_return)}</td>
            <td className="num">{pct(h[k].predicted_volatility_annualized)}</td>
            <td className="num">{pct(h[k].value_at_risk_95)}</td>
            <td className="num">{h[k].prob_loss_beyond_threshold !== null ? `${pct(h[k].prob_loss_beyond_threshold)} (< ${pct(h[k].loss_threshold, 0)})` : NA}</td>
          </tr>))}</tbody>
      </table></div>
      <div className="stat-grid">
        <Stat label="Risk score" value={NA} hint="The system does not define a composite risk score; the individual risk forecasts are shown instead" />
        <Stat label="P(|1-day move| > 2%)" value={pct(fc.large_move_probability_1d)} />
        {fc.regime_probabilities && Object.entries(fc.regime_probabilities).map(([k, v]) => (
          <Stat key={k} label={`Regime ${fc.regime_horizon_days}d: ${k.replace("_", " ")}`} value={pct(v)} />
        ))}
      </div>
      <Note tone="warn">Out-of-sample tests found no reliable skill for direction or return forecasts; treat P(up) and the
        expected return as close to the historical base rate. Volatility and downside forecasts did show skill. See Model Performance.</Note>
      {fc.data_quality_warnings.length > 0 && (
        <details className="small"><summary>{fc.data_quality_warnings.length} data-quality warnings</summary>
          <ul>{fc.data_quality_warnings.map((w) => <li key={w}>{w}</li>)}</ul></details>
      )}
    </Card>
  );
}

function Drivers({ fc }: { fc: Forecast }) {
  return (
    <div className="table-scroll"><table>
      <thead><tr><th>Forecast</th><th>Model</th><th>Feature categories</th><th>Training data end</th></tr></thead>
      <tbody>{Object.entries(fc.tasks).map(([t, v]) => (
        <tr key={t}><td>{taskLabel(t)}</td><td>{modelLabel(v.model)}</td><td>{v.categories.join(", ")}</td><td>{v.train_end}</td></tr>
      ))}</tbody>
    </table>
    <p className="muted small">Feature coverage at forecast time: {Object.entries(fc.feature_coverage).map(([k, v]) => `${k} ${pct(v, 0)}`).join(" · ")}</p>
    </div>
  );
}

function RecentPredictions({ mid, avail }: { mid: string; avail: AvailablePrediction[] }) {
  const period = avail.some((a) => a.period === "holdout") ? "holdout" : "dev";
  const sel = avail.find((a) => a.period === period && a.task === "volatility_20" && a.selected)
    ?? avail.find((a) => a.period === period && a.task === "volatility_20" && !a.reference);
  const q = useApi(sel ? () => api.predictions({ market: mid, task: sel.task, model: sel.model, feature_set: sel.feature_set, period }) : null,
    [mid, sel?.model, sel?.feature_set, period]);
  const rows = (q.data?.rows ?? []).slice(-8).reverse();
  return (
    <Card title="Recent prediction history" subtitle={sel ? `20-day volatility · ${modelLabel(sel.model)} · ${period === "dev" ? "development" : "holdout"} walk-forward` : undefined}
      actions={<Link to="/backtest" className="btn ghost">All history →</Link>}>
      {!sel && <Empty>No stored volatility predictions yet.</Empty>}
      {q.loading && <Loading />}
      {rows.length > 0 && (
        <table><thead><tr><th>Predicted on</th><th>Resolved on</th><th className="num">Predicted vol</th><th className="num">Actual vol</th></tr></thead>
          <tbody>{rows.map((r) => (
            <tr key={r.prediction_date}><td>{r.prediction_date}</td><td>{r.target_date}</td>
              <td className="num fc-text">{pct(r.predicted_vol)}</td><td className="num">{pct(r.actual_vol)}</td></tr>))}</tbody></table>
      )}
    </Card>
  );
}

function historyOption(rows: PriceRow[], fc: Forecast | null, p: Palette): EChartsOption {
  const dates = rows.map((r) => r.date);
  const closes = rows.map((r) => r.close);
  const last = closes[closes.length - 1];
  const extra: string[] = [];
  const mid: (number | null)[] = [];
  const lo: (number | null)[] = [];
  const band: (number | null)[] = [];
  if (fc) {
    const hs = Object.keys(fc.horizons_trading_days).map(Number).sort((a, b) => a - b);
    hs.forEach((h) => {
      const f = fc.horizons_trading_days[String(h)];
      if (f.expected_log_return === null || f.return_uncertainty_sd === null) return;
      extra.push(`t+${h} sessions`);
      const m = last * Math.exp(f.expected_log_return);
      const l = last * Math.exp(f.expected_log_return - 1.645 * f.return_uncertainty_sd);
      const u = last * Math.exp(f.expected_log_return + 1.645 * f.return_uncertainty_sd);
      mid.push(m); lo.push(l); band.push(u - l);
    });
  }
  const pad = (arr: (number | null)[], anchor: number | null) => [...Array(dates.length - 1).fill(null), anchor, ...arr];
  const ax = axisStyle(p);
  return {
    ...baseOption(p),
    legend: { top: 0, textStyle: { color: p.text }, data: ["Observed close", "Forecast 90% band", "Forecast (expected)"] },
    grid: { left: 64, right: 16, top: 30, bottom: 48 },
    xAxis: { type: "category", data: [...dates, ...extra], ...ax },
    yAxis: { type: "value", scale: true, ...ax },
    dataZoom: [{ type: "inside", start: 50 }, { type: "slider", height: 18, bottom: 6, start: 50 }],
    series: [
      { name: "Observed close", type: "line", data: closes, showSymbol: false, lineStyle: { color: p.actual, width: 1.6 }, itemStyle: { color: p.actual } },
      ...(extra.length ? [
        { name: "band-low", type: "line" as const, data: pad(lo, last), stack: "band", lineStyle: { opacity: 0 }, showSymbol: false, tooltip: { show: false } },
        { name: "Forecast 90% band", type: "line" as const, data: pad(band, 0), stack: "band", lineStyle: { opacity: 0 }, showSymbol: false, areaStyle: { color: p.band } },
        { name: "Forecast (expected)", type: "line" as const, data: pad(mid, last), lineStyle: { color: p.forecast, type: "dashed" as const, width: 2 }, itemStyle: { color: p.forecast } },
      ] : []),
    ],
  };
}
