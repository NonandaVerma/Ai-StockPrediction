# Market Risk Lab — frontend

React 18 + TypeScript + Vite 4 + ECharts 5 UI for the `marketpred` backend. It only displays data the
backend has stored. It never generates, interpolates or shifts predictions in the browser.

## Run

```bash
# 1. backend (from the repository root)
.venv/Scripts/python -m uvicorn marketpred.api.app:app --port 8000      # Windows; use .venv/bin/python elsewhere
# 2. frontend
cd frontend
npm install
cp .env.example .env          # VITE_API_BASE_URL=http://127.0.0.1:8000
npm run dev                   # http://127.0.0.1:5173
```

Other scripts: `npm run build` (type-check + production build to `dist/`), `npm test` (Vitest),
`npm run typecheck`. Requires Node ≥ 14.18 (tested with 14.18.0).

## Pages

| Route | Page |
|---|---|
| `#/backtest` | **Historical Backtesting & Prediction Comparison** (primary): filters, synchronised actual/prediction charts, accuracy summary with baseline test, by-horizon table, error analysis, per-prediction table with CSV export, methodology |
| `#/` | Market overview: latest observed data, latest forecast, price history with forecast band, recent predictions |
| `#/risk` | Risk alerts, forecast vs realised volatility, drawdowns, downside probability, regime probabilities, stress episodes |
| `#/factors` | Category ablation evidence (CIs, Holm-corrected p), astrology protocol, feature catalogue |
| `#/performance` | Model comparison vs baselines, holdout results, trading simulation, VaR backtest |
| `#/status` | Data sources, stale data, run/version metadata, holdout-use log, limitations |

## Where the numbers come from

* **Historical predictions** are the walk-forward out-of-sample records stored by `marketpred evaluate` /
  `holdout`. Each was made by a model trained only on purged data that ended before the prediction date
  (the `train_end` column shows exactly when). The UI labels development-period results as
  selection-biased when used to choose the best model; the holdout is the unbiased test.
* **Predicted prices** are derived from return forecasts as `close_at_prediction · exp(predicted log return)`
  and plotted at the target date (`t + h` sessions on the exchange calendar).
* **Metrics** come from `GET /api/evaluation` (the backend's evaluation code). The only metrics computed
  in the browser are volatility MAE/RMSE in volatility points, derived directly from the returned rows.
* Unavailable items (feature importance, attribution, composite risk score) are shown as N/A. Their
  required API contracts are in `../docs/API.md`.

## Structure

```
src/api/        typed client (client.ts) and response interfaces (types.ts)
src/components/ layout, theme/market context, ECharts wrapper (synchronised groups), data table, UI primitives
src/lib/        date-range validation, formatting, statistics, CSV (unit-tested in lib.test.ts)
src/pages/      one file per page
```
