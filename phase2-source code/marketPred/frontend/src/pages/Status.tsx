import { api, API_BASE } from "../api/client";
import { DataTable } from "../components/DataTable";
import { useApp } from "../components/context";
import { Badge, Card, ErrorBox, Loading, Stat } from "../components/ui";
import { useApi } from "../lib/useApi";

const LIMITATIONS = [
  "Market data are end-of-day closes from a free, unofficial source (Yahoo); they are delayed and may contain vendor errors.",
  "The S&P 500 price index excludes dividends; total returns are higher.",
  "S&P 500 open prices before 2007 are unreliable in the source data and are not used by any model.",
  "NFCI and Shiller data are latest-vintage downloads of revised series (possible revision look-ahead); Shiller data end in 2024.",
  "Macro release delays are conservative constants, not exact historical release calendars.",
  "Weather data are reanalysis (NASA POWER), not real-time observations.",
  "No licensed news archive: article-level sentiment is not part of the evaluated models.",
  "The holdout period (2021 onward) is a single ~6-year window; its results carry wide uncertainty.",
];

export function StatusPage() {
  const { market } = useApp();
  const mid = market!.id;
  const st = useApi(() => api.status(mid), [mid]);
  const health = useApi(() => api.health(), []);
  return (
    <div className="page">
      <div className="page-head"><div><h1>Data Sources &amp; System Status</h1></div></div>
      <div className="grid-2">
        <Card title="Service">
          <div className="stat-grid">
            <Stat label="API" value={health.data ? <Badge tone="good">online</Badge> : health.error ? <Badge tone="bad">offline</Badge> : "…"} hint={API_BASE} />
            <Stat label="API version" value={health.data?.version ?? "—"} />
            <Stat label="Latest price data" value={st.data?.price_coverage.end ?? "—"} hint="Last session in the ingested data" />
            <Stat label="Forecast generated" value={st.data ? (st.data.forecast_available ? "yes" : "no") : "—"} />
          </div>
        </Card>
        <Card title="Model run">
          {st.loading && <Loading />}
          {st.data && (st.data.run ? (
            <div className="stat-grid">
              <Stat label="Run" value={<code className="small">{st.data.run.name}</code>} />
              <Stat label="Created (UTC)" value={st.data.run.created_utc.replace("T", " ").slice(0, 16)} />
              <Stat label="Dataset version" value={<code>{st.data.run.dataset_version}</code>} />
              <Stat label="Config fingerprint" value={<code>{st.data.run.config_fingerprint}</code>} />
              <Stat label="Stages" value={Object.entries(st.data.run.stages_complete).map(([k, v]) => (
                <Badge key={k} tone={v ? "good" : "neutral"}>{k}{v ? " ✓" : " …"}</Badge>))} />
              <Stat label="Holdout evaluations" value={String(st.data.run.holdout_evaluations.length)}
                hint="Each use of the final test period is logged; more than one would weaken its validity" />
            </div>
          ) : <p className="muted">No experiment run yet for this market.</p>)}
        </Card>
      </div>
      {st.error && <ErrorBox error={st.error.message} />}
      {st.data && (
        <>
          <Card title="Data sources" subtitle="Most recent ingestion of each source">
            <DataTable rows={st.data.sources as unknown as Record<string, unknown>[]} pageSize={20} columns={[
              { key: "source", label: "Source" }, { key: "dataset", label: "Dataset" },
              { key: "status", label: "Status", render: (r) => <Badge tone={r.status === "ok" ? "good" : "bad"}>{String(r.status)}</Badge> },
              { key: "fetched_at", label: "Last update (UTC)", render: (r) => String(r.fetched_at).replace("T", " ").slice(0, 16) },
              { key: "min_date", label: "From" }, { key: "max_date", label: "To" },
              { key: "rows", label: "Rows", align: "right" }, { key: "message", label: "Message" },
            ]} />
          </Card>
          <Card title="Stale or ended data" subtitle="Features whose last available value is more than 10 days old (treated as missing, never filled)">
            {st.data.stale_features.length ? (
              <DataTable rows={st.data.stale_features as unknown as Record<string, unknown>[]} pageSize={15} columns={[
                { key: "feature", label: "Feature" }, { key: "category", label: "Category" },
                { key: "last_valid", label: "Last value" }, { key: "days_stale", label: "Days stale", align: "right" },
              ]} />
            ) : <p className="muted">All features are current.</p>}
          </Card>
        </>
      )}
      <Card title="Known dataset limitations">
        <ul className="method">{LIMITATIONS.map((l) => <li key={l}>{l}</li>)}</ul>
        <p className="muted small">Full details: docs/DATA_SOURCES.md and docs/REMAINING_WORK.md.</p>
      </Card>
    </div>
  );
}
