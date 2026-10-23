# Methodology

## 1. Decision time and point-in-time data

* **Decision timestamp** for session *t*: the official exchange close of *t* in UTC
  (`calendar.decision_times`, from `pandas_market_calendars`; handles DST and early closes, e.g.
  13:00 ET on the day after Thanksgiving).
* Every external observation carries `available_at` (UTC). A value is used at decision time *T* only if
  `available_at <= T` (`data/pit.align`, an as-of join). Consequences:
  * VIX (16:15 ET), futures and the dollar index (~17:00 ET) and Treasury yields (18:00 ET) published
    after the 16:00 equity close are used from the **next** session.
  * Monthly macro data enter after conservative lags (CPI +35 days after month end, unemployment +55,
    consumer sentiment +1, Shiller earnings +120). Weekly NFCI enters the following Wednesday 08:30 ET.
    CFTC positions (as of Tuesday) enter the following Monday.
  * For a non-US market (NIFTY 50, close 15:30 IST = 10:00 UTC), US data become usable only after their
    US publication time, normally on the next Indian session. No manual lagging is needed.
* **Backfills** (late publication of an older period) never replace a newer observation. **Revisions**
  of the newest observation replace it from their publication time. With multiple vintages (e.g.
  ALFRED), transforms are recomputed from the snapshot known at each publication time (`pit.derive`).
* **Staleness:** values older than `max_staleness_days` become missing instead of being carried forward.
* **Missing data are never fabricated.** Providers' missing values are dropped at ingestion; features
  are NaN until a source starts; models handle NaN natively (trees) or impute inside the training
  pipeline (median and missing-indicator, fitted on training rows only).

## 2. Targets (exact definitions)

Let *C_i* be the close of session *i*, *r_i* = ln(*C_i*/*C_{i-1}*) and *h* the horizon in sessions.
All labels are formed from sessions *i+1..i+h* only and evaluated at the close of *i+h*.

| Task | Definition | Horizons |
|---|---|---|
| direction_h | 1{ln(C_{i+h}/C_i) > 0}; output = probability | 1, 5, 20 |
| return_h | ln(C_{i+h}/C_i) (log return; percentage = e^x - 1) | 1, 5, 20 |
| volatility_h | ln sqrt(252/h · Σ_{k=1..h} r_{i+k}²) (annualised realised vol, log) | 5, 20 |
| downside_h | 1{ln(C_{i+h}/C_i) < c_h}, c_5 = -2%, c_20 = -5% | 5, 20 |
| large_move_1 | 1{\|ln(C_{i+1}/C_i)\| > 2%} (event risk proxy) | 1 |
| regime_20 | high_vol if RV_20 > 25%; else bull if R_20 > +2%; bear if R_20 < -2%; else sideways | 20 |

Tail risk is also reported as a 95% Value-at-Risk derived from volatility forecasts,
VaR = 1.645 · σ̂_h (normal approximation), backtested with the Kupiec (1995) test on non-overlapping windows.
Scheduled-event risk (e.g. around FOMC/CPI dates) needs a historical release calendar, which is not
yet ingested; `large_move_1` is the generic large-move version.

## 3. Validation design

* **Development period** (model and feature selection): out-of-sample predictions from an expanding
  window, refit every 252 sessions, test dates 2000-01-01 .. 2020-12-31.
* **Purging and embargo:** training row *i* is admissible for a test block starting at *s* only if
  *i + h < s − 5*. Development test rows whose label window reaches the holdout start are dropped.
* **Final holdout:** 2021-01-01 to the end of data, used once after selection. Every use is recorded in
  `artifacts/holdout_lock.json`; re-use requires `--force` and is logged.
* **Training-only preprocessing:** winsorisation quantiles, imputation medians, scaling, hyper-parameter
  choice (small grid for linear models) and Platt calibration are all estimated inside each training
  window. Tuning and calibration use the last 20% of the window, separated by a purge gap.
* **No random splits** anywhere.

## 4. Models

| Family | Models | Notes |
|---|---|---|
| Benchmarks | climatology (training frequency), historical mean, random-walk volatility | the "no skill" references |
| Statistical | AR(1) (`ARIMA(1,0,0)`), HAR-RV (Corsi 2009, range-based daily term), GARCH(1,1)-t | GARCH parameters estimated per window, filtered forward with fixed parameters |
| Regularised linear | logistic (L2), ridge, elastic net | in-window tuning of the penalty |
| Trees | LightGBM, XGBoost, random forest (CatBoost optional) | fixed, conservative hyper-parameters; Platt-calibrated for probabilities |
| Deep learning | not implemented | ~9k daily observations of one index; no evidence it would help; deferred until simpler models are beaten |

## 5. Evaluation and statistics

* Binary: log loss (primary), Brier and Brier skill score, ROC-AUC, ECE and reliability table,
  balanced accuracy, precision, recall, F1 (threshold 0.5).
* Returns: RMSE (primary), MAE, Campbell-Thompson out-of-sample R² against the training-window mean,
  correlation, sign hit rate.
* Volatility: MSE of log RV (primary), QLIKE, Mincer-Zarnowitz regression, bias.
* Regime: multiclass log loss (primary), Brier, accuracy, balanced accuracy.
* Comparisons: paired Diebold-Mariano tests on per-observation losses with Newey-West HAC
  (lag ≥ h−1) and Harvey-Leybourne-Newbold correction; 95% circular-block-bootstrap CIs (block ≥ 20,
  ≥ h) for mean loss differences and AUC differences; Holm correction across the eight single-category
  additions per task and model; regime breakdown by ex-ante 63-day realised volatility
  (<12%, 12-20%, >20%).
* Economic value: long/cash simulation from `direction_1` probabilities (threshold 0.5, pre-specified),
  5 bps cost + 2 bps slippage per unit turnover, cash earns the 3-month T-bill; Sharpe, Sortino,
  max drawdown, turnover, exposure, and results by volatility regime, against buy-and-hold.

## 6. Selection rule (pre-registered in code, `training/experiments.Experiment.select`)

For each task: the *primary* ablation model is the one with the lower development loss on the
price-only set. A category **survives** if adding it to the price baseline lowers the primary model's
loss with a Holm-adjusted DM p-value < 0.05. The combined feature set is price + surviving categories.
The *selected* model is the configuration with the lowest development loss in the model comparison.
The selection is written to `selection.json` before the holdout is evaluated.

## 7. Reproducibility

* Raw downloads are kept as timestamped Parquet snapshots; the fetch log (rows, content hash, date
  range, status) is in `data/metadata.db`.
* The feature matrix has a content hash (`dataset_version`) and a provenance/coverage JSON.
* Each run directory stores the full resolved config, its fingerprint, the dataset version, every
  out-of-sample prediction (Parquet) and all result tables. `docs/RESULTS.md` is generated from them.
* Seeds are fixed (`project.seed`); LightGBM/XGBoost use deterministic seeds and fixed thread counts,
  but small floating-point differences across machines or library versions are possible.
