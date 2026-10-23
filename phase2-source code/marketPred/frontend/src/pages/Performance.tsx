import { useMemo, useState } from "react";
import { api } from "../api/client";
import type { ComparisonRow, TradingResult } from "../api/types";
import { DataTable } from "../components/DataTable";
import { useApp } from "../components/context";
import { Badge, Card, Empty, ErrorBox, Field, Loading, Note, Segmented, Select } from "../components/ui";
import { featureSetLabel, modelLabel, num, pct, pval, taskLabel } from "../lib/format";
import { useApi } from "../lib/useApi";

const PRIMARY: Record<string, string> = { binary: "log_loss", multiclass: "log_loss", regression: "rmse", volatility: "mse_log" };
const EXTRA: Record<string, string[]> = {
  binary: ["auc", "brier", "ece", "accuracy", "balanced_accuracy", "precision", "recall", "f1"],
  multiclass: ["brier", "accuracy", "balanced_accuracy"],
  regression: ["r2_oos", "mae", "hit_rate", "corr"],
  volatility: ["qlike", "mae_log", "mz_r2", "bias_log"],
};

export function PerformancePage() {
  const { market } = useApp();
  const mid = market!.id;
  const comp = useApi(() => api.comparison(mid), [mid]);
  const hold = useApi(() => api.holdout(mid), [mid]);
  const [period, setPeriod] = useState<"dev" | "holdout">("dev");
  const trading = useApi(() => api.trading(mid, period), [mid, period]);
  const varb = useApi(() => api.varBacktest(mid, period), [mid, period]);
  const tasks = useMemo(() => Array.from(new Set((comp.data ?? []).map((r) => String(r.task)))), [comp.data]);
  const [task, setTask] = useState("volatility_20");
  const t = tasks.includes(task) ? task : tasks[0];
  const rows = (comp.data ?? []).filter((r) => r.task === t);
  const kind = String(rows[0]?.kind ?? "binary");
  const lossCol = PRIMARY[kind];
  const v = market!.run?.validation;

  return (
    <div className="page">
      <div className="page-head"><div>
        <h1>Model Performance</h1>
        <p className="muted">All numbers are out-of-sample. Development: walk-forward {v?.dev_start} → {v?.holdout_start}, expanding
          window refit every {v?.retrain_every_days} sessions, purge = horizon + {v?.embargo_days}-session embargo. Holdout:
          {" "}{v?.holdout_start} onward, evaluated once after model selection.</p>
      </div></div>

      <Card title="Model comparison (development period)" actions={
        <Field label="Task"><Select value={t ?? ""} onChange={setTask} options={tasks.map((x) => ({ value: x, label: taskLabel(x) }))} /></Field>}>
        {comp.loading && <Loading />}
        {comp.error && (comp.error.status === 404 ? <Empty>Results appear when the development evaluation finishes.</Empty> : <ErrorBox error={comp.error.message} />)}
        {rows.length > 0 && (
          <DataTable rows={rows as Record<string, unknown>[]} pageSize={30} csvName={`comparison_${t}.csv`} columns={[
            { key: "model", label: "Model", render: (r) => modelLabel(String(r.model)) },
            { key: "feature_set", label: "Features", render: (r) => featureSetLabel(String(r.feature_set)) },
            { key: "n", label: "n", align: "right" },
            { key: lossCol, label: `${lossCol} (primary)`, align: "right", render: (r) => num(r[lossCol], 4) },
            ...EXTRA[kind].map((k) => ({ key: k, label: k, align: "right" as const, render: (r: Record<string, unknown>) => num(r[k], 4) })),
            { key: "vs_ref", label: "vs baseline (DM p)", align: "right",
              value: (r) => refP(r as ComparisonRow), render: (r) => refCell(r as ComparisonRow) },
          ]} />
        )}
        <p className="muted small">Baselines: climatology (base rate) for probabilities, the historical mean for returns,
          random-walk volatility for volatility. Directional accuracy alone is not evidence of skill: compare log loss and AUC with the baseline.</p>
      </Card>

      <Card title="Final holdout (untouched until after selection)">
        {hold.loading && <Loading />}
        {hold.error && (hold.error.status === 404 ? <Empty>The one-time holdout evaluation has not been run yet.</Empty> : <ErrorBox error={hold.error.message} />)}
        {hold.data && <DataTable rows={hold.data as Record<string, unknown>[]} pageSize={40} csvName="holdout.csv" columns={[
          { key: "task", label: "Task", render: (r) => taskLabel(String(r.task)) },
          { key: "role", label: "Role" },
          { key: "model", label: "Model", render: (r) => modelLabel(String(r.model)) },
          { key: "categories", label: "Categories" },
          { key: "n", label: "n", align: "right" },
          { key: "loss", label: "Primary loss", align: "right", value: (r) => r.log_loss ?? r.mse_log ?? r.rmse,
            render: (r) => num(r.log_loss ?? r.mse_log ?? r.rmse, 4) },
          { key: "vs_ref_rel_improvement", label: "vs baseline", align: "right", render: (r) => pct(r.vs_ref_rel_improvement, 2) },
          { key: "vs_ref_p_value", label: "DM p", align: "right", render: (r) => pval(r.vs_ref_p_value) },
        ]} />}
      </Card>

      <div className="row gap"><Segmented value={period} onChange={setPeriod} options={[
        { value: "dev", label: "Development period" }, { value: "holdout", label: "Holdout period" }]} /></div>

      <Card title="Trading simulation (1-day direction, long when P(up) > 0.5, else T-bills)"
        subtitle="Net of 5 bps cost + 2 bps slippage per unit turnover; price index without dividends">
        {trading.loading && <Loading />}
        {trading.error && <Empty>Not available for this period yet.</Empty>}
        {trading.data && <TradingTable data={trading.data} />}
        <Note>Sharpe ratio and drawdown are reported only for this pre-specified, cost-aware walk-forward simulation. A higher
          Sharpe than buy-and-hold here would still need to survive the holdout and multiple-testing considerations.</Note>
      </Card>

      <Card title="Value-at-Risk backtest (95%, from volatility forecasts)" subtitle="Kupiec proportion-of-failures test on non-overlapping windows; expected violation rate 5%">
        {varb.loading && <Loading />}
        {varb.error && <Empty>Not available for this period yet.</Empty>}
        {varb.data && <table><thead><tr><th>Horizon</th><th>Model</th><th className="num">n</th><th className="num">Violations</th>
          <th className="num">Rate</th><th className="num">Kupiec p</th><th>Verdict</th></tr></thead>
          <tbody>{varb.data.map((r, i) => (
            <tr key={i}><td>{String(r.horizon)}d</td><td>{modelLabel(String(r.model))}</td><td className="num">{String(r.n)}</td>
              <td className="num">{String(r.violations)}</td><td className="num">{pct(r.rate, 1)}</td><td className="num">{pval(r.p_value)}</td>
              <td>{Number(r.p_value) < 0.05 ? <Badge tone="bad">coverage rejected</Badge> : <Badge tone="good">consistent with 5%</Badge>}</td></tr>))}</tbody></table>}
      </Card>
    </div>
  );
}

function refP(r: ComparisonRow): number | null {
  const k = Object.keys(r).find((x) => x.startsWith("vs_") && x.endsWith("_p_value") && r[x] !== null);
  return k ? Number(r[k]) : null;
}

function refCell(r: ComparisonRow) {
  const pk = Object.keys(r).find((x) => x.startsWith("vs_") && x.endsWith("_p_value") && r[x] !== null);
  if (!pk) return <span className="muted">baseline</span>;
  const ik = pk.replace("_p_value", "_loss_improvement");
  const better = Number(r[ik]) > 0;
  const p = Number(r[pk]);
  return <span>{pval(p)} {p < 0.05 ? <Badge tone={better ? "good" : "bad"}>{better ? "better" : "worse"}</Badge> : null}</span>;
}

function TradingTable({ data }: { data: Record<string, TradingResult> }) {
  const entries = Object.entries(data);
  const bh = entries[0]?.[1].buy_and_hold;
  return (
    <div className="table-scroll"><table>
      <thead><tr><th>Strategy</th><th className="num">Ann. return</th><th className="num">Ann. vol</th><th className="num">Sharpe</th>
        <th className="num">Sortino</th><th className="num">Max DD</th><th className="num">Exposure</th><th className="num">Turnover / yr</th></tr></thead>
      <tbody>
        {bh && <tr className="ref-row"><td>Buy and hold</td><td className="num">{pct(bh.ann_return)}</td><td className="num">{pct(bh.ann_vol)}</td>
          <td className="num">{num(bh.sharpe, 2)}</td><td className="num">{num(bh.sortino, 2)}</td><td className="num">{pct(bh.max_drawdown)}</td>
          <td className="num">100%</td><td className="num">0</td></tr>}
        {entries.map(([k, v]) => {
          const [m, fs] = k.split("|");
          return <tr key={k}><td>{modelLabel(m)} · {featureSetLabel(fs)}</td><td className="num">{pct(v.strategy.ann_return)}</td>
            <td className="num">{pct(v.strategy.ann_vol)}</td><td className="num">{num(v.strategy.sharpe, 2)}</td>
            <td className="num">{num(v.strategy.sortino, 2)}</td><td className="num">{pct(v.strategy.max_drawdown)}</td>
            <td className="num">{pct(v.exposure, 0)}</td><td className="num">{num(v.turnover_per_year, 1)}</td></tr>;
        })}
      </tbody></table></div>
  );
}
