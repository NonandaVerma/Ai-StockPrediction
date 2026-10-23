# Limitations, missing inputs and remaining work

Status key: **Implemented + tested** = code exists and automated tests or end-to-end runs exercised it.
**Implemented, untested on real data** = code and unit tests exist but no suitable real dataset was
available. **Not implemented** = design only or nothing.

## Status of spec items

| Item | Status |
|---|---|
| Ingestion (11 keyless sources), raw snapshots, metadata DB, retries, cache, rate limit, validation | Implemented + tested (ran end to end 2026-10-16) |
| Point-in-time alignment, staleness, backfill/revision handling | Implemented + tested (unit tests incl. synthetic vintages) |
| ALFRED vintage ingestion (`fred` connector) | Implemented, **untested on real data** (needs `FRED_API_KEY`; keyless FRED unreachable from the test network) |
| Feature categories A-I (price, macro, valuation, cross-asset, news indices, weather, CFTC, astro, calendar) | Implemented + evaluated |
| Article-level news NLP (dedup, novelty, event typing, sentiment, point-in-time daily aggregation) | Implemented, **untested on real data** (no licensed archive); unit-tested on synthetic headlines |
| Targets 1-6 | Implemented + tested; event risk is the generic large-move version (no scheduled-event calendar yet) |
| Baselines, AR(1), HAR, GARCH, logistic/ridge/elastic net, LightGBM, XGBoost, RF | Implemented + evaluated |
| CatBoost | Supported in the factory if installed; **not installed or evaluated** |
| LSTM / TCN / Transformers | **Not implemented** (deliberately deferred; see REPORT) |
| Walk-forward, purging, embargo, holdout lock, in-window tuning and calibration | Implemented + tested |
| Ablation with DM/HAC, block bootstrap, Holm, regime breakdown | Implemented + run |
| Trading simulation with costs; VaR backtest | Implemented + run |
| Forecast records, forecast log, drift (PSI) | Implemented + run once (no live history yet, so no forecast-vs-outcome results) |
| FastAPI service (`/api/*` for the UI) | Implemented; unit-tested for health/404, smoke-tested against real run artifacts |
| Web UI (React + TypeScript, `frontend/`) | Implemented; type-checked, lib unit tests, visually checked via headless-Chrome screenshots of each page. Per-feature importance, attribution and a composite risk score are shown as N/A (contracts in docs/API.md) |
| Streamlit dashboard | Not implemented (superseded by the React UI) |

## Missing data / credentials that would most improve the system

1. **Intraday data** (5-minute S&P 500 returns) for true realised variance: better volatility targets
   and HAR inputs. Sources: TAQ, Refinitiv, or the Oxford-Man realised library (discontinued 2022).
2. **ALFRED vintages** (`FRED_API_KEY`): remove the revision look-ahead in NFCI; add credit spreads
   (BAA10Y), breakevens and exact historical release dates for every macro series.
3. **Options data** (implied skew, risk-neutral moments, put/call volumes): OptionMetrics or CBOE DataShop.
4. **Licensed news archive** with original timestamps (Dow Jones, Refinitiv, RavenPack) to evaluate the
   NLP pipeline; **consensus forecasts** for macro surprises (Bloomberg/Refinitiv).
5. **Point-in-time fundamentals and index membership** (Compustat Snapshot, CRSP, I/B/E/S) for
   single-stock tasks without survivorship bias.
6. **Indian market data:** NSE FII/DII flows, India VIX term structure, RBI/MOSPI macro with release
   dates, and an exchange-sourced price history (Yahoo lacks Muhurat sessions and the 2025-02-01
   Budget-day session).
7. Total-return index (dividends) for the trading simulation.

## Known methodological limitations

* Release lags are conservative constants, not historical release calendars.
* NFCI and Shiller data are latest vintages (revision leakage possible, flagged in metadata).
* Shiller valuation data end in 2024-06, so valuation features are missing for most of the holdout.
* NASA POWER is reanalysis, not a real-time observation record.
* Weather and astronomy features use the exchange city / Earth-centred conventions described in the
  feature modules; other conventions were deliberately *not* searched (to avoid data mining).
* Eclipse detection is an approximation (one false-positive solar date in 2017-2025; penumbral lunar
  eclipses partly included; the tiny 2024-09-18 partial lunar eclipse missed).
* The holdout (2021-) is a single period of ~5.8 years; its conclusions carry wide uncertainty.
* Multiple-testing correction is applied within each task/model family; with 12 tasks the
  experiment-wide false-positive risk is higher than the nominal 5%.

## Suggested next steps (in priority order)

1. Paper-trade: run `ingest → build-features → predict` daily, then reconcile `forecast_log.csv` with
   outcomes (`monitoring.drift.reconcile_forecast_log`).
2. Add intraday realised variance and ALFRED vintages; re-run the volatility and macro ablations.
3. Extend to weather-sensitive markets (corn/wheat, natural gas) where the weather mechanism is credible,
   using archived forecasts rather than observations.
4. Only if a non-linear model shows a robust, significant gain: try sequence models (LSTM/TCN) on
   intraday-derived inputs with the same walk-forward protocol.
5. Dashboard (Streamlit) reading the API.
