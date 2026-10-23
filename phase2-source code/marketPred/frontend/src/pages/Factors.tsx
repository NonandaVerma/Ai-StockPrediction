import type { EChartsOption } from "echarts";
import { useMemo, useState } from "react";
import { api } from "../api/client";
import type { AblationRow, FeatureInfo } from "../api/types";
import { DataTable } from "../components/DataTable";
import { axisStyle, baseOption, EChart, Palette, usePalette } from "../components/EChart";
import { useApp } from "../components/context";
import { Badge, Card, Empty, ErrorBox, Field, Loading, Note, Select } from "../components/ui";
import { featureSetLabel, modelLabel, NA, num, pct, pval, taskLabel } from "../lib/format";
import { useApi } from "../lib/useApi";

const CATEGORIES: Record<string, { title: string; desc: string; status?: string }> = {
  price: { title: "Technical indicators & price behaviour", desc: "Returns, momentum, realised/downside vol, ranges, distance from highs, RSI, volume" },
  valuation: { title: "Fundamentals & valuation", desc: "Index earnings and dividend yields (Shiller), lagged 120 days" },
  macro: { title: "Macroeconomic indicators", desc: "Treasury curve, financial conditions (NFCI), CPI, unemployment, consumer sentiment" },
  news: { title: "News, politics & geopolitics", desc: "Daily Economic Policy Uncertainty and Geopolitical Risk indices (news-based)",
    status: "Article-level news sentiment is implemented but not evaluated: no licensed historical archive" },
  cross_asset: { title: "Cross-market & derivatives", desc: "VIX level/term structure, variance risk premium, dollar, gold, oil, copper/gold, credit ETFs, small caps" },
  weather: { title: "Weather & climate", desc: "Exchange-city cloud cover, clearness, temperature anomalies, precipitation (NASA POWER)" },
  behavior: { title: "Market positioning & alternative data", desc: "CFTC futures positioning (speculators, small traders, open interest)" },
  astro: { title: "Astrology / astronomical cycles (experimental)", desc: "Lunar phase, Mercury retrograde, eclipses, Jupiter–Saturn angle, hard aspects" },
  calendar: { title: "Calendar & seasonality (controls)", desc: "Day of week, turn of month, holidays, January, Nov–Apr" },
};

export function FactorsPage() {
  const { market } = useApp();
  const pal = usePalette();
  const mid = market!.id;
  const feats = useApi(() => api.features(mid), [mid]);
  const abl = useApi(() => api.ablation(mid), [mid]);
  const tasks = useMemo(() => Array.from(new Set((abl.data ?? []).map((r) => r.task))), [abl.data]);
  const [task, setTask] = useState("volatility_20");
  const t = tasks.includes(task) ? task : tasks[0];
  const models = Array.from(new Set((abl.data ?? []).filter((r) => r.task === t).map((r) => r.model)));
  const [model, setModel] = useState("");
  const m = models.includes(model) ? model : models[0];
  const rows = (abl.data ?? []).filter((r) => r.task === t && r.model === m && r.feature_set !== "price");

  return (
    <div className="page">
      <div className="page-head"><div>
        <h1>Features &amp; Factors</h1>
        <p className="muted">Which information categories improve out-of-sample forecasts, tested one at a time against the
          price-only baseline with identical folds and tuning budgets.</p>
      </div></div>

      <Note tone="warn">Improvement in out-of-sample loss shows that a category <em>helps forecast</em> in this sample. It does
        not show that the factor <em>causes</em> market moves. Categories with no significant improvement — including astrology
        and weather — are not used by the selected models unless they pass the Holm-corrected test.</Note>

      <Card title="Category ablation results" actions={
        <div className="row gap">
          <Field label="Task"><Select value={t ?? ""} onChange={setTask} options={tasks.map((x) => ({ value: x, label: taskLabel(x) }))} /></Field>
          <Field label="Model"><Select value={m ?? ""} onChange={setModel} options={models.map((x) => ({ value: x, label: modelLabel(x) }))} /></Field>
        </div>}>
        {abl.loading && <Loading />}
        {abl.error && (abl.error.status === 404
          ? <Empty>Ablation results are not available yet (the development evaluation is still running or has not been run).</Empty>
          : <ErrorBox error={abl.error.message} />)}
        {rows.length > 0 && (
          <>
            <EChart height={360} ariaLabel="Relative loss improvement by category" option={ablationOption(rows, pal)} />
            <p className="muted small">Bars: relative reduction in out-of-sample loss vs. price-only (positive = better). Whiskers: 95%
              block-bootstrap CI (in loss units, scaled). Filled bars are significant after Holm correction.</p>
            <DataTable rows={rows as unknown as Record<string, unknown>[]} pageSize={20} csvName={`ablation_${t}_${m}.csv`} columns={[
              { key: "feature_set", label: "Feature set", render: (r) => featureSetLabel(String(r.feature_set)) },
              { key: "n_features", label: "Features", align: "right" },
              { key: "test_coverage_added", label: "Coverage of added data", align: "right", render: (r) => pct(r.test_coverage_added, 0) },
              { key: "rel_improvement", label: "Loss improvement", align: "right", render: (r) => pct(r.rel_improvement, 2) },
              { key: "ci", label: "95% CI (loss units)", value: (r) => r.ci_low, render: (r) => `[${num(r.ci_low, 4)}, ${num(r.ci_high, 4)}]` },
              { key: "p_value", label: "DM p", align: "right", render: (r) => pval(r.p_value) },
              { key: "holm_p", label: "Holm p", align: "right", render: (r) => (r.holm_p === null || r.holm_p === undefined ? "—" : pval(r.holm_p)) },
              { key: "significant_improvement", label: "Verdict", render: (r) => verdict(r as unknown as AblationRow) },
            ]} />
          </>
        )}
      </Card>

      <Card title="Astrology protocol" subtitle="Four comparisons on identical out-of-sample dates, as pre-registered">
        {rows.length > 0 ? <AstroTable rows={(abl.data ?? []).filter((r) => r.task === t && r.model === m)} /> : <Empty>Available after the ablation completes.</Empty>}
      </Card>

      <Card title="Individual feature importance / attribution">
        <Empty>{NA}: the backend does not compute per-feature importance or attribution yet, so none is displayed (and none is
          estimated in the browser). The contract for adding it is <code>GET /api/feature-importance</code> in docs/API.md.</Empty>
      </Card>

      <Card title="Feature catalogue" subtitle="Every feature with its source, transform and data coverage">
        {feats.loading && <Loading />}
        {feats.data && <Catalogue feats={feats.data} />}
      </Card>
    </div>
  );
}

function verdict(r: AblationRow) {
  if (r.holm_p === null || r.holm_p === undefined) {
    return r.p_value !== null && r.p_value < 0.05
      ? <Badge tone={(r.loss_improvement ?? 0) > 0 ? "good" : "bad"}>{(r.loss_improvement ?? 0) > 0 ? "better" : "worse"} (uncorrected)</Badge>
      : <Badge>no difference</Badge>;
  }
  if (r.significant_improvement) return <Badge tone="good">improves</Badge>;
  if ((r.holm_p ?? 1) < 0.05 && (r.loss_improvement ?? 0) < 0) return <Badge tone="bad">hurts</Badge>;
  return <Badge>no significant effect</Badge>;
}

function AstroTable({ rows }: { rows: AblationRow[] }) {
  const sets = ["calendar_only", "astro_only", "price+calendar", "price+astro", "price+calendar+astro"];
  const by = new Map(rows.map((r) => [r.feature_set, r]));
  const lossKey = rows[0]?.kind === "volatility" ? "mse_log" : rows[0]?.kind === "regression" ? "rmse" : "log_loss";
  return (
    <table><thead><tr><th>Model input</th><th className="num">Out-of-sample loss</th><th className="num">vs price-only</th><th className="num">DM p</th></tr></thead>
      <tbody>
        <tr><td>Price only (standard financial model)</td><td className="num">{num(by.get("price")?.[lossKey], 4)}</td><td className="num">—</td><td className="num">—</td></tr>
        {sets.map((s) => {
          const r = by.get(s);
          return <tr key={s}><td>{featureSetLabel(s)}</td><td className="num">{num(r?.[lossKey], 4)}</td>
            <td className="num">{pct(r?.rel_improvement, 2)}</td><td className="num">{pval(r?.p_value)}</td></tr>;
        })}
      </tbody></table>
  );
}

function Catalogue({ feats }: { feats: FeatureInfo[] }) {
  const cats = Array.from(new Set(feats.map((f) => f.category)));
  return (
    <div className="cat-grid">
      {cats.map((c) => {
        const fs = feats.filter((f) => f.category === c);
        const info = CATEGORIES[c] ?? { title: c, desc: "" };
        return (
          <details key={c} className="cat">
            <summary><strong>{info.title}</strong> <span className="muted">· {fs.length} features</span>
              <div className="muted small">{info.desc}</div>
              {info.status && <div className="small warn-text">{info.status}</div>}</summary>
            <table><thead><tr><th>Feature</th><th>Source</th><th>Transform</th><th>Coverage</th><th>Available</th></tr></thead>
              <tbody>{fs.map((f) => (
                <tr key={f.feature}><td><code>{f.feature.split("__")[1]}</code></td><td>{f.source ?? NA}</td><td>{f.transform ?? NA}</td>
                  <td className="num">{pct(f.coverage, 0)}</td><td>{f.first_valid ?? NA} → {f.last_valid ?? NA}</td></tr>))}</tbody></table>
          </details>
        );
      })}
    </div>
  );
}

function ablationOption(rows: AblationRow[], p: Palette): EChartsOption {
  const ax = axisStyle(p);
  const names = rows.map((r) => featureSetLabel(r.feature_set));
  const scale = (r: AblationRow) => (r.loss_improvement && r.rel_improvement ? r.rel_improvement / r.loss_improvement : 0);
  return {
    ...baseOption(p), legend: { show: false }, tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    grid: { left: 160, right: 24, top: 16, bottom: 32 },
    yAxis: { type: "category", data: names, inverse: true, ...ax },
    xAxis: { type: "value", ...ax, axisLabel: { ...ax.axisLabel, formatter: (v: number) => `${(v * 100).toFixed(0)}%` } },
    series: [
      { name: "Relative loss improvement", type: "bar", data: rows.map((r) => ({
          value: r.rel_improvement ?? 0,
          itemStyle: { color: (r.rel_improvement ?? 0) >= 0 ? p.up : p.down, opacity: r.significant_improvement || ((r.holm_p ?? 1) < 0.05) ? 0.9 : 0.35 },
        })) },
      { name: "95% CI", type: "custom", renderItem: (_params, api) => {
          const i = api.value(0) as number;
          const lo = api.coord([api.value(1), i]);
          const hi = api.coord([api.value(2), i]);
          return { type: "line", shape: { x1: lo[0], y1: lo[1], x2: hi[0], y2: hi[1] }, style: { stroke: p.text, lineWidth: 1.2 } };
        }, encode: { x: [1, 2], y: 0 }, data: rows.map((r, i) => [i, (r.ci_low ?? 0) * scale(r), (r.ci_high ?? 0) * scale(r)]), z: 10 },
    ],
  };
}
