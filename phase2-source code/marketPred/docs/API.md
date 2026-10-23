# Backend API contract (used by `frontend/`)

Base URL: `VITE_API_BASE_URL` (default `http://127.0.0.1:8000`). All endpoints are read-only `GET`s over
stored artifacts; no request trains a model or generates a prediction. CORS origins:
`MARKETPRED_CORS_ORIGINS` (default `http://localhost:5173,http://127.0.0.1:5173`).
Errors: `404` = data not available yet (e.g. evaluation still running), `422` = request cannot be
evaluated (e.g. fewer than 20 predictions in the window). Body: `{"detail": "<reason>"}`.

Interactive schema: `http://127.0.0.1:8000/docs`.

## Implemented

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | `{status, version}` |
| `GET /api/markets` | Markets with built features: id, name, ticker, calendar, timezone, currency, `price_coverage {start,end}`, `tasks[] {task, kind, target, horizon}`, `run` (latest run metadata incl. validation settings and completed stages) |
| `GET /api/prices?market&start&end` | Observed end-of-day OHLCV + realised 1-day log return (`return_1d`). `open_reliable_from` flags dates before which opens are unreliable |
| `GET /api/predictions/available?market` | Inventory of stored walk-forward prediction sets: `{period, task, model, feature_set, kind, horizon, selected, reference}` |
| `GET /api/predictions?market&task&model&feature_set&period&start&end` | One row per stored out-of-sample prediction, with `prediction_date`, `target_date` (session *t+h* on the exchange calendar), `close_at_prediction`, `actual_close_target`, `actual_return`, `fold`, `train_end`, plus kind-specific fields (see below). `meta` states the method, `leakage_free: true`, `retrospective_fit: false`, number of refits, calibration and the price-conversion formula |
| `GET /api/evaluation?…same filters…` | Metrics for the window computed with `marketpred.evaluation` (same code as the reports), price-level MAE/RMSE/MAPE for return forecasts, baseline comparison (Diebold-Mariano p, block-bootstrap CI), per-volatility-regime metrics |
| `GET /api/run/{comparison_dev, ablation_dev, holdout, selection, trading_dev, trading_holdout, var_backtest_dev, var_backtest_holdout}?market` | Result tables of the latest run |
| `GET /api/features?market` | Feature catalogue: category, source, transform, coverage, first/last valid date |
| `GET /api/forecast/latest?market` | Latest forecast record (`marketpred predict`) |
| `GET /api/status?market` | Latest fetch per source, price coverage, feature dataset version, stale features, run metadata, holdout-use log |

Kind-specific prediction fields:

* **regression** (return_h): `predicted_return` (log), `predicted_price = close_at_prediction · exp(predicted_return)`, `predicted_direction`, `actual_direction`, `price_error`, `price_error_pct`, `return_error`
* **volatility** (volatility_h): `predicted_vol`, `actual_vol` (annualised), `vol_error`, `vol_error_pct`
* **binary** (direction_h, downside_h, large_move_1): `probability`, `actual_event`, `predicted_event` (P > 0.5), `prob_error`; direction adds `predicted_direction`/`actual_direction`
* **multiclass** (regime_20): `prob_bull|bear|sideways|high_vol`, `actual_regime`, `predicted_regime`

Only predictions whose outcome is observed are stored; unresolved live forecasts come from
`/api/forecast/latest` and `artifacts/predictions/forecast_log.csv`.

## Required for features the UI shows as N/A (not implemented)

| Proposed endpoint | Needed for | Backend work |
|---|---|---|
| `GET /api/feature-importance?market&task&model&feature_set&period` → `[{feature, category, importance, importance_ci_low, importance_ci_high, method}]` | Per-feature importance on the Factors page | Compute permutation importance on each walk-forward test block (out-of-sample, so not inflated by fitting) and store per fold |
| `GET /api/attribution/latest?market&task` → `[{feature, value, contribution}]` | "Important factors" for the latest forecast | SHAP/TreeSHAP for tree models, coefficients × standardised values for linear models, stored with the forecast record |
| `GET /api/risk-score/history?market` | A composite risk score | First define and validate a score (e.g. calibrated probability of a >5% 20-day drawdown); none exists today |
| `GET /api/forecast/history?market` → reconciled live forecasts | Paper-trading vs backtest comparison | Expose `monitoring.drift.reconcile_forecast_log` once daily forecasts have accumulated |
| `GET /api/predictions?…&horizon=10` | 10-day horizon | Add `10` to `targets.horizons` and re-run `evaluate`; the UI lists whatever horizons the backend stores |
