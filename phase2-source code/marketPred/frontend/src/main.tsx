import React from "react";
import ReactDOM from "react-dom/client";
import { HashRouter, Route, Routes } from "react-router-dom";
import { AppProvider } from "./components/context";
import { Layout } from "./components/Layout";
import { BacktestPage } from "./pages/Backtest";
import { FactorsPage } from "./pages/Factors";
import { OverviewPage } from "./pages/Overview";
import { PerformancePage } from "./pages/Performance";
import { RiskPage } from "./pages/Risk";
import { StatusPage } from "./pages/Status";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <AppProvider>
      <HashRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/backtest" element={<BacktestPage />} />
            <Route path="/risk" element={<RiskPage />} />
            <Route path="/factors" element={<FactorsPage />} />
            <Route path="/performance" element={<PerformancePage />} />
            <Route path="/status" element={<StatusPage />} />
          </Route>
        </Routes>
      </HashRouter>
    </AppProvider>
  </React.StrictMode>,
);
