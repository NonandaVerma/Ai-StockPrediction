import { NavLink, Outlet } from "react-router-dom";
import { API_BASE } from "../api/client";
import { useApp } from "./context";
import { ErrorBox, Loading, Select } from "./ui";

const NAV = [
  { to: "/backtest", label: "Historical Backtesting", primary: true },
  { to: "/", label: "Market Overview" },
  { to: "/risk", label: "Risk Analysis" },
  { to: "/factors", label: "Features & Factors" },
  { to: "/performance", label: "Model Performance" },
  { to: "/status", label: "Data & Status" },
];

export function Layout() {
  const { markets, market, setMarketId, theme, toggleTheme, loading, error } = useApp();
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden>◆</span>
          <span>Market Risk Lab</span>
          <span className="brand-sub">research system · not investment advice</span>
        </div>
        <div className="topbar-actions">
          {market && (
            <Select ariaLabel="Market" value={market.id} onChange={setMarketId}
              options={markets.map((m) => ({ value: m.id, label: `${m.name} (${m.ticker})` }))} />
          )}
          <button type="button" className="btn ghost" onClick={toggleTheme} aria-label="Toggle colour theme">
            {theme === "dark" ? "☀ Light" : "☾ Dark"}
          </button>
        </div>
      </header>
      <nav className="nav" aria-label="Main">
        {NAV.map((n) => (
          <NavLink key={n.to} to={n.to} end={n.to === "/"}
            className={({ isActive }) => `${isActive ? "active" : ""} ${n.primary ? "primary" : ""}`}>
            {n.label}
          </NavLink>
        ))}
      </nav>
      <main className="main">
        {loading && <Loading label="Connecting to backend…" />}
        {error && <ErrorBox error={error} hint={<>Backend URL: <code>{API_BASE}</code> (set <code>VITE_API_BASE_URL</code>).</>} />}
        {!loading && !error && !market && <ErrorBox error="The backend reports no markets with built features." hint="Run marketpred ingest and build-features." />}
        {market && <Outlet />}
      </main>
      <footer className="footer muted">
        Observed data are delayed end-of-day closes, not live quotes. Historical predictions are stored walk-forward
        out-of-sample forecasts. Nothing here is investment advice.
      </footer>
    </div>
  );
}
