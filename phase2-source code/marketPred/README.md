# marketpred — multi-factor market prediction research system

A modular Python system for forecasting **direction, returns, volatility, downside risk, large moves and
market regimes** of an equity index, built around **point-in-time data** and **leakage-free
walk-forward validation**. It tests nine candidate feature categories (prices, macro, fundamentals,
cross-asset, news/geopolitics, weather, investor positioning, astronomy, calendar) under one protocol
and keeps a category only if it improves out-of-sample forecasts after multiple-testing correction.

**First market:** S&P 500 price index (`^GSPC`), daily decisions at the NYSE close, horizons 1/5/20
sessions. **Extension:** NIFTY 50 (`configs/nifty50.yaml`).

> Research software. Nothing here is investment advice. The headline finding is that direction and
> return forecasts showed **no reliable out-of-sample skill**, while volatility and risk forecasts did.
> See [docs/REPORT.md](docs/REPORT.md).

## Headline results (out-of-sample, S&P 500)

Final holdout 2021-01 → 2026-10 (~1,450 forecasts per task, evaluated once after selection on 2000-2020 walk-forward):

| Forecast | Holdout result vs baseline |
|---|---|
| Large 1-day move (\|r\| > 2%) | −18.9% log loss, AUC 0.81 (p < 0.001) |
| 20-day / 5-day volatility | −38% / −41% MSE vs random-walk vol; beats HAR and GARCH (p < 0.001) |
| 20-day regime (bull/bear/sideways/high-vol) | −10.6% log loss (p < 0.001) |
| Downside (P(loss > 2% in 5d, > 5% in 20d)) | price-only logistic −3.4% / −6.6% (p = 0.012 / 0.028); the selected all-category variant showed no skill |
| Direction 1/5/20d, returns 1/5/20d | **no skill** (no model beats the base rate / historical mean) |

No feature category (macro, valuation, cross-asset, news/geopolitics, weather, CFTC positioning,
calendar, astrology) improved any task over price-based features after Holm correction.
Details: [docs/REPORT.md](docs/REPORT.md); full tables: [docs/RESULTS.md](docs/RESULTS.md).

## Contents

| Path | What |
|---|---|
| `src/marketpred/data/` | connectors (Yahoo, Treasury, BLS, Chicago Fed, UMich, Shiller, EPU, GPR, CFTC, NASA POWER, optional FRED/ALFRED), storage, validation, point-in-time engine (`pit.py`) |
| `src/marketpred/features/` | one module per category + `news_nlp.py` (event extraction for licensed archives) |
| `src/marketpred/targets/` | target definitions |
| `src/marketpred/models/` | benchmarks, AR(1), HAR, GARCH, linear, LightGBM/XGBoost/RF, Platt calibration |
| `src/marketpred/training/` | purged walk-forward engine, experiment protocol (comparison, ablation, selection, holdout) |
| `src/marketpred/evaluation/` | metrics, Diebold-Mariano / bootstrap / Holm / Kupiec, report generator |
| `src/marketpred/backtesting/` | cost-aware long/cash simulation |
| `src/marketpred/prediction/` | structured forecast records |
| `src/marketpred/monitoring/` | feature drift (PSI), forecast-vs-outcome reconciliation |
| `src/marketpred/api/` | optional read-only FastAPI service |
| `configs/` | `default.yaml` (S&P 500, all timing rules), `nifty50.yaml` |
| `tests/` | 49 unit/integration/leakage tests (offline) |
| `frontend/` | React + TypeScript UI (Historical Backtesting & Prediction Comparison, overview, risk, factors, performance, status) — see `frontend/README.md` |
| `docs/` | research review, data sources, methodology, generated results, final report, remaining work |

## Setup

Requires Python 3.10-3.12 (tested on 3.11.4, Windows 10). No API keys are needed.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt         # exact versions used for the reported results
pip install -e .                        # installs the `marketpred` command
cp .env.example .env                    # optional: FRED_API_KEY, BLS_API_KEY
```

Behind a TLS-inspecting proxy, the OS certificate store is used automatically (`truststore`).

## Run the full workflow

```bash
python -m pytest                        # 49 offline tests
python -m marketpred ingest             # ~2 min; raw snapshots + processed point-in-time series in data/
python -m marketpred build-features     # ~10 s; data/features/SPX_features.parquet (+ provenance JSON)
python -m marketpred evaluate           # development walk-forward: model comparison, ablation, selection
python -m marketpred holdout            # ONE-TIME evaluation on the untouched 2021+ holdout
python -m marketpred report             # regenerate docs/RESULTS.md from the latest run
python -m marketpred predict            # forecast record for the latest session -> artifacts/predictions/
python -m marketpred monitor            # feature drift report -> artifacts/monitoring/
uvicorn marketpred.api.app:app          # API for the UI (http://127.0.0.1:8000/docs)
cd frontend && npm install && npm run dev   # UI at http://127.0.0.1:5173
```

`evaluate` takes ~1 hour on 12 cores (`compute.n_jobs` in the config; `--quick` for a ~15-minute
smoke run in `artifacts/quick_runs/`). Runs are resumable: `evaluate --run-dir <dir>` reuses saved
predictions. Another market: add `--config configs/nifty50.yaml` to every command.

## Configuration

Everything that affects results is in `configs/default.yaml`: market and calendar, per-source
publication rules (`release_lag_days`, `release_time`, `max_staleness_days`), target horizons and
thresholds, validation dates, purge/embargo, model lists, hyper-parameters, tuning grids, bootstrap
settings and trading costs. Override with `--config my.yaml` (deep-merged). Credentials are read only
from environment variables / `.env` (never stored in code or configs).

## Outputs

* `data/raw/<source>/<dataset>/<timestamp>.parquet` — immutable download snapshots; `data/metadata.db` — fetch log
* `data/processed/series/*.parquet` — long-format series with `obs_date`, `value`, `available_at` (UTC)
* `artifacts/runs/<run>/` — manifest (config + dataset version), every out-of-sample prediction, result tables
* `artifacts/holdout_lock.json` — record of every holdout evaluation
* `artifacts/predictions/` — forecast JSON records and `forecast_log.csv` (for paper-trading reconciliation)

## Documentation

* [docs/RESEARCH.md](docs/RESEARCH.md) — literature review, evidence grading, category priorities
* [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) — fields, coverage, timing, revisions, licensing, limitations
* [docs/METHODOLOGY.md](docs/METHODOLOGY.md) — point-in-time rules, target definitions, validation, statistics
* [docs/RESULTS.md](docs/RESULTS.md) — generated tables from the latest run
* [docs/REPORT.md](docs/REPORT.md) — interpretation, test results, limitations
* [docs/REMAINING_WORK.md](docs/REMAINING_WORK.md) — missing data/credentials, unimplemented items
* [docs/API.md](docs/API.md) — API contract used by the UI (implemented + required future endpoints)
