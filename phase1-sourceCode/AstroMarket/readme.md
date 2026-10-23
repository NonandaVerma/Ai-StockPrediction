# Market Prediction

A market forecasting and research platform. It forecasts both stock/index **direction** and
**volatility** from many data categories: price and technical data, macro and global markets
(S&P 500, Brent, exchange-appropriate FX, India VIX), geopolitical risk, India weather, the
feature-selection catalog categories, and planetary positions (Vedic or Western astrology). Every
category is tested out of sample against market-only baselines and a locked holdout, with strict
anti-overfitting checks - as of the last full test pass, **volatility forecasting (benchmark/VIX
context) is the one capability with confirmed out-of-sample evidence**; direction prediction has
not cleared that bar for any feature set or algorithm tried so far (see `src/vol_engine.py` and the
Model Comparison -> Volatility tab).

> Research project only. Not investment advice. Do not trade real money on its output.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py                                    # dashboard
pytest -q                                               # unit tests (offline)
pytest -q -m "not slow"                                 # skip the positive-control test
```

This project is UI-only: every workflow (forecasting, model comparison, backtesting, the risk
pipeline's volatility comparisons) is a page in the Streamlit dashboard. There is no separate
command-line pipeline to run.

## Market Prediction dashboard

The default dashboard section. It works with any stock or index the data provider supports
(e.g. `^NSEI`, `^BSESN`, `RELIANCE.NS`, `^GSPC`, `AAPL`).
The stock picker is on every page; Forecast Date and Generate Forecast appear only on Home, the page
that shows the forecast. Astrology is one data category: on Home its cards (current sky, similar past
skies, what it means) are grouped under "Category: Astrology".

| Screen | What it shows |
|---|---|
| Market Overview | Latest price with an honest freshness badge (real-time / delayed / end-of-day), intraday-to-yearly charts, the sky now (astronomical vs astrological columns), upcoming events |
| Historical Analysis | Prices with astro conditions highlighted, per-session price + sky table, condition statistics with FDR correction, event studies (returns around new moons, stations, ingresses...) |
| Future Forecast | Pick next N sessions, a custom date or a date range, plus an as-of date (pick a past date to replay a forecast point-in-time). Shows direction, expected return, 80% interval, volatility/risk, validated confidence, evidence label, historical analogues, all models side by side, limitations |
| Model Comparison | Walk-forward metrics of baseline, market-only, astro-only, hybrid and every catalog-category model on identical dates; paired tests of whether astro adds value; optional permutation test. Its **Volatility** tab compares a HAR benchmark against HAR + each data category and shows a live "as of today" expected-volatility forecast for whichever category is statistically validated |
| Backtesting Results | Cost-adjusted equity curves, predicted vs actual, rolling accuracy, stored backtest runs and stored forecasts with realised outcomes |
| Actual vs Predicted | Candlestick comparison of real prices against predictions made without those prices (any data category combination, any algorithm), plus accuracy/coverage metrics |

### How forecasts are made
* **Target:** after the as-of close, skip `gap` sessions and predict the log return over the next
  `horizon` sessions. Next-N, custom date and custom range all reduce to this one definition.
  Calendar dates are mapped to real sessions (weekends/holidays shift to the next session and are explained).
* **Models** (`src/forecast_models.py`): naive no-change, naive historical mean, market-only
  (L1 logistic + time-series-CV ridge, LightGBM), astro-only Model A (L1, feature selection) and LightGBM,
  hybrid market+astro (linear, LightGBM, XGBoost), one "market + category" pair per catalog category, and
  a kitchen-sink "all categories" model (kept mainly as an overfitting warning sign - see Model Comparison).
* **Hyperparameter tuning & early stopping:** LightGBM/XGBoost always fit with early stopping on an internal,
  time-ordered validation tail (never the outer test block) instead of a fixed tree count - the single
  biggest accuracy lever found so far. A `TrainSettings.tune` flag (UI checkbox: "Tune hyperparameters
  (slower)" on Future Forecast and Model Comparison) additionally runs a small nested walk-forward search
  over a few hand-picked hyperparameter sets; off by default since it's ~4-8x slower per fit.
* **Validation** (`src/forecast_engine.py`): expanding walk-forward, never shuffled, embargo of gap+horizon
  sessions, all models on identical dates. Significance is tested on non-overlapping windows.
* **Evidence labels:** `validated` only if accuracy beats always-predicting-the-majority with p<0.05 and
  AUC > 0.5 out of sample; otherwise `experimental`; `insufficient` with too little data.
  Confidence is the out-of-sample hit rate of similar past predictions, shown only with >= 40
  independent cases. Intervals come from out-of-sample errors.
* **Point-in-time:** the engine only ever receives prices up to the as-of session
  (`tests/test_forecast.py::test_forecast_uses_only_point_in_time_data` corrupts later prices and checks the
  forecast is unchanged). Exchange calendars and planetary positions are known in advance, so they are used.

### Volatility forecasting (`src/vol_engine.py`)
Separate from direction prediction: forecasts *how much* a stock will move, not which way. A HAR
(Heterogeneous Autoregressive) volatility benchmark enriched with category 05 (benchmark-index
correlation + India VIX) beats the plain HAR baseline by 12-27% (QLIKE loss), confirmed significant
with Holm-corrected Diebold-Mariano tests across ^BSESN/^NSEI/individual NSE stocks and 5/10/20-session
horizons - and the improvement held up (actually strengthened) on a locked 2-year holdout never touched
during model selection. This is the one capability in the project with real out-of-sample confirmation;
direction prediction has not cleared that bar for any feature set or algorithm tried.
* `vol_engine.live_forecast()` / `forecast_service.run_vol_forecast()`: fits on all available history and
  predicts the as-of session's expected volatility (point estimate + 80% range), as opposed to the
  walk-forward backtest comparison on the Volatility tab.
* Surfaced on **Home**'s "Expected Volatility" tile and as a live callout on **Model Comparison -> Volatility**.
  Restricted to NSE/BSE listings: category 05 builds for every exchange, but the significance + holdout
  testing behind the "validated" claim was only ever run on Indian symbols (where it also includes India
  VIX, not just benchmark correlation) - showing the same claim for an untested market would overstate
  the evidence, so the tile silently falls back to its plain-language estimate for other exchanges.

### Data, storage, credentials
* **Market data** (`src/market_providers.py`): provider-independent interface with retries, exponential backoff,
  throttling and timeouts. `yahoo` (default, no key, unofficial feed, never labelled real-time),
  `alphavantage` (daily bars, needs env var `ALPHAVANTAGE_API_KEY`), `synthetic` (offline testing).
  Missing data raises an error and is never filled in with made-up values.
* **Exchanges** (`src/exchanges.py`): NSE, BSE, NYSE, NASDAQ, LSE with local session hours; UTC conversion
  through `zoneinfo` (DST-aware). Past sessions come from the data; future sessions need
  `data/calendars/<EXCHANGE>_holidays.csv` (see `data/calendars/README.md`).
* **Ephemeris** (`src/ephemeris.py`, astronomical) vs **astrology** (`src/astrology.py`, interpretation):
  Sun to Pluto plus mean/true nodes; longitude, latitude, distance, speed, declination; signs, nakshatras,
  padas, tithis, phases, aspects with orbs, ingresses, stations, lunations, eclipse seasons. The model-facing
  astrology feature set is deliberately smaller than the raw sky table: Neptune/Uranus/Pluto (near-collinear
  with each other over any one instrument's history) and the near-static longitude of Jupiter/Saturn/the
  lunar node are excluded from training features (their retrograde flags are kept), and each planet pair
  contributes one separation angle instead of five near-duplicate "orb" columns - cut the flagship hybrid
  model's astrology inputs from 539 to 38 features with no information loss, which measurably helps the
  linear models (collinearity hurts logistic/ridge regression; tree models already ignore the duplicates).
* **Database:** SQLite at `data/astromarket.db`, schema in `migrations/*.sql` (applied automatically by
  `src/db.py`). Tables: `instruments`, `data_sources`, `price_bars`, `data_freshness`, `planetary_positions`,
  `astro_events`, `model_versions`, `backtest_runs`, `predictions`. Unique keys prevent duplicate imports,
  runs and forecasts. Raw prices/positions are stored separately from derived results.
* **Help texts:** every tooltip in the UI comes from the `help_texts` table (key = control id or table
  column name). Edit them in the **Glossary** tab ("Edit help texts") or directly in the database.

## Configuration

Everything lives in `config.yaml`: ticker, dates, zodiac and ayanamsa, observation time,
aspect orb, horizon, holdout length, folds, permutations, FDR level, trading cost,
`forecast.tune_hyperparams` (default `false` - the UI checkbox overrides this per-session instead
of needing a config edit).

For maximum ephemeris precision download the Swiss Ephemeris `.se1` files and set
`astro.ephemeris_path`; otherwise the built-in Moshier ephemeris (about 1 arcsecond) is used.

## Exploratory notebooks

`notebooks/feature_correlation.ipynb` - correlation/redundancy analysis per catalog category (which
astrology/technical/volatility features are near-duplicates of each other). Exploratory only; it never
touches `src/` and any feature changes it motivates are made separately, by hand, in `catalog_features.py`
or `astrology.py`. Needs `ipykernel` (run the project's `.venv` as the notebook kernel) and `seaborn`.

## Project layout

```
market-prediction/
├── config.yaml
├── app.py                 # Streamlit dashboard (the only entry point)
├── migrations/            # SQLite schema + help texts
├── src/
│   ├── config.py                       # settings
│   ├── db.py, repository.py            # storage
│   ├── exchanges.py                    # exchange hours, DST, trading calendar
│   ├── market_providers.py, market_service.py
│   ├── ephemeris.py, astrology.py      # astronomy vs interpretation
│   ├── analysis.py                     # condition stats, event studies, analogues
│   ├── catalog_features.py             # one feature group per catalog category
│   ├── forecast_models.py, training.py, forecast_engine.py, forecast_service.py
│   ├── hindcast.py, vol_engine.py      # actual vs predicted, volatility target + live forecast
│   ├── ui_forecast.py, ui_dashboard.py, ui_components.py   # dashboard
│   ├── astro_features.py, features.py, validation.py, evaluate.py
│   └── risk_data.py, risk_external.py, risk_targets.py, risk_eval.py
├── notebooks/             # exploratory feature-correlation analysis (never touches src/)
├── tests/
├── data/{raw,processed,calendars}/
└── outputs/
```
