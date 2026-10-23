# Final report — S&P 500 multi-factor prediction system (v0.1, 2026-10-16)

All numbers below come from `artifacts/runs/20261016T114001Z_dfcee00527b8` (dataset version
`dfcee00527b8`), are reproduced in full in [RESULTS.md](RESULTS.md), and are out-of-sample.
Development = walk-forward 2000-01 → 2020-12 (~5,280 forecasts per task); holdout = 2021-01 → 2026-10
(~1,450 forecasts per task), evaluated **once** after selection (logged in `artifacts/holdout_lock.json`).

## 1. Headline findings

1. **Risk is forecastable; direction and returns are not.** This matches the pre-registered
   expectations in [RESEARCH.md](RESEARCH.md) §4 and the literature (Welch & Goyal 2008; Corsi 2009).
2. **No additional data category earned its place.** Under the pre-registered rule (Holm-corrected
   Diebold-Mariano test for the task's primary model), no category — macro, valuation, cross-asset,
   news/geopolitics, weather, positioning, calendar or astrology — significantly improved any of the 12
   tasks over price-based features.
3. **The holdout confirmed this the hard way.** Where development selection narrowly preferred the
   all-category model (5-day volatility, both downside tasks), the price-only version of the same model
   was better on the holdout; for downside risk, the all-category model lost its significance entirely.

## 2. Holdout results (final, unbiased)

| Task | Selected (on dev) | Holdout loss vs baseline | p (DM) | Verdict |
|---|---|---|---|---|
| Large move, \|1d\| > 2% | logistic, price | −18.9% log loss, AUC 0.81 | < 0.001 | **Skill** |
| Volatility 20d | ridge, price | −38% MSE (log vol); also beats HAR (−30%) and GARCH (−17%) vs RW | < 0.001 | **Skill** |
| Volatility 5d | ridge, all categories | −41% MSE (price-only ridge: −49%) | < 0.001 | **Skill** (price-only better) |
| Regime 20d | logistic, price | −10.6% log loss | < 0.001 | **Skill** |
| Downside 5d (loss > 2%) | logistic, all categories | −0.5% | 0.85 | No skill (price-only logistic: −3.4%, p = 0.012) |
| Downside 20d (loss > 5%) | logistic, all categories | −2.7% | 0.63 | No skill (price-only logistic: −6.6%, p = 0.028) |
| Direction 1d | logistic, price | +0.04% (worse), AUC 0.50 | 0.81 | **No skill** |
| Direction 5d / 20d | climatology (base rate) | — | — | Models did not beat the base rate on dev |
| Return 1d / 5d / 20d | AR(1) | R² −0.37% / +0.26% / −0.01% | 0.52 / 0.28 / 0.91 | **No skill** |

Baselines: climatology for probabilities, historical mean for returns, random-walk (trailing realised)
volatility. Every comparison is paired on identical dates. CIs and all other models are in RESULTS.md §6.

**Directional accuracy is misleading here.** Holdout 20-day "accuracy" is 66.4% — identical to the
fraction of up periods (66.4%), because the models (like the base rate) almost always forecast "up".
Calibrated log loss, Brier score and AUC are the meaningful measures, and they show no skill.

## 3. Development-period results (used for selection)

* Volatility: ridge/HAR/LightGBM beat random-walk volatility by 26-43% (MSE of log vol, p < 0.001);
  HAR has the best QLIKE at 20 days. LightGBM did not beat the regularised linear models.
* Large moves (AUC 0.85, −26% log loss) and regimes (−12.6%) — strong; downside 5d modest (−3.7%, p = 0.002);
  downside 20d none.
* Direction: best 1d model indistinguishable from the base rate (p = 0.80); for 5d/20d every model was
  worse than the base rate (20d logistic significantly worse, p = 0.001) — the classic overfitting signature.
* Returns: AR(1) 20d R² = 0.13% (p = 0.014) — economically negligible and it vanished on the holdout.
* Tree ensembles (LightGBM, XGBoost, random forest) were never the best model for any task.

## 4. Feature-category ablation (all evaluated categories)

Each category was added to the price baseline with the same models, folds and tuning budget
(RESULTS.md §3). Summary across 12 tasks × 2 models:

| Category | Evidence | Decision |
|---|---|---|
| Macro & rates | Mostly **worse** (e.g. −7.7% for 20d vol, −4.6% for regimes): level features (rates, CPI) are non-stationary and extrapolate badly | Rejected |
| Valuation (Shiller E/P, D/P) | No significant effect; data end in 2024 | Rejected |
| Cross-asset (VIX, credit, dollar, commodities) | Mixed for volatility: +1.0% / +1.5% (5d, ridge / LightGBM) and +0.6% (20d LightGBM), but −6.1% for 20d ridge; none significant (all Holm p ≥ 0.37). A direct HAR+log-VIX regression shows VIX carries information over a 4-variable HAR, but the richer price feature set appears to capture most of it, and the other cross-asset features add noise | Rejected under the rule |
| News & geopolitics (EPU, GPR) | +1.3% for regimes (p = 0.026, Holm p = 0.18) | Rejected |
| Weather (NYC) | Significant only for LightGBM on 5d downside (+0.7%, Holm p = 0.003) — the weaker model for that task; not for the primary model | Rejected; treated as a likely false positive of model-shopping |
| Positioning (CFTC) | No significant improvement; often worse | Rejected |
| Calendar controls | No significant effect | Rejected |
| **Astrology** | Astro-only and calendar-only models both lose heavily to price for risk tasks (e.g. 5d vol: −120% to −137%), and adding astro to price+calendar makes volatility forecasts significantly worse (−4% to −15%, p ≤ 0.001); for direction both are indistinguishable from each other and from price; price+astro never improves on price after correction (largest: +1.9% for 20d returns with ridge, p = 0.011, Holm p = 0.084, on a task with no skill at all) | **Rejected** — no evidence beyond calendar controls |

Survival of a category only for a weaker model, or before multiple-testing correction, was the
recurring pattern. The pre-registered "primary model + Holm" rule prevented those from being retained.

## 5. Economic value and risk diagnostics

* **Long/cash strategy** (1d direction, P(up) > 0.5, 7 bps per unit turnover, cash earns T-bills):
  * Development: logistic/price Sharpe 0.32 vs buy-and-hold 0.25, max drawdown −47% vs −57%
    (89% invested). The all-category logistic variant had a lower drawdown (−34%) with 30× annual turnover.
  * Holdout: logistic/price Sharpe 0.68 vs 0.65, max drawdown −20.5% vs −25.4% (93% invested).
  * These differences are small, rest on ~7-11% of days spent in cash, and were not tested for
    significance; the underlying direction forecasts have no statistically detectable skill. Treat as
    **not demonstrated**.
* **VaR (95%, normal approximation from volatility forecasts)**: under-covers at 5 days for HAR/ridge-type
  forecasts (dev 7-8%, holdout 8-9% violations; Kupiec p < 0.05). GARCH-t VaR passes in both periods
  (dev 4.1%, holdout 5.5%). Volatility *level* forecasts are better with ridge/HAR, but tail quantiles
  need fat-tailed distributions — use GARCH-t or a t/empirical quantile for VaR.

## 6. Recommendations

1. Use the system for **volatility, large-move, regime and downside-risk forecasting** with price-based
   features; do not use its direction/return outputs for trading.
2. Promote the price-only variants for 5d volatility and downside (holdout evidence). This decision
   was informed by the holdout, so it must be validated on new data (paper trading) before relying on it.
3. Replace the normal VaR mapping with a Student-t or empirical-quantile mapping.
4. Highest-value data additions: intraday realised variance, options-implied information, ALFRED vintages
   (see [REMAINING_WORK.md](REMAINING_WORK.md)).

## 7. Tests and verification actually performed

| Check | Result |
|---|---|
| Python test suite (`pytest`): PIT alignment, DST/early closes, vintages, backfills, targets, feature formulas, causality perturbation tests, purging/embargo, holdout isolation, planted-leak detection, GARCH causality, metrics, DM/bootstrap/Holm/Kupiec, backtest accounting, NLP pipeline, API | **49 passed, 0 failed** |
| Frontend (`vitest`): date-range validation, statistics, formatting, CSV | **11 passed, 0 failed**; `tsc --noEmit` clean; production build OK |
| End-to-end: ingest (11 sources) → features → dev evaluation (470 walk-forward prediction sets) → selection → holdout → report → forecast → drift monitor | Completed 2026-10-16 |
| Manual checks: VIX alignment (value at *t* = VIX close of *t−1*); feature hash unchanged after refactor; eclipse catalogue vs. published dates; calendar checks found genuine provider gaps | Done (see conversation log / DATA_SOURCES.md) |

Bugs found and fixed during validation: label-aligned series construction that silently dropped BLS
unemployment (regression test added); tiny values rendering as 0.0000 in the UI.

## 8. Extension market: NIFTY 50 (same protocol)

Run `artifacts/nifty50/runs/20261016T121653Z_c7b0ee6ea409`; development 2015-01 → 2021-12, holdout
2022-01 → 2026-10 (~1,170 forecasts per task), evaluated once. Categories: price, calendar, US macro,
cross-asset (India VIX + global), news indices, Mumbai weather, astrology (no valuation/CFTC data for India).

| Task | Holdout vs baseline | p |
|---|---|---|
| Large move 1d (logistic, price) | −24.7% log loss | < 0.001 |
| Volatility 5d (ridge, price) | −41.5% MSE vs random-walk vol | < 0.001 |
| Volatility 20d (HAR, price) | −29.1% | 0.020 |
| Direction 1/5/20d, returns, downside 5/20d, regime | no significant improvement (all-category selections for returns/regime did not hold up) | ≥ 0.06 |

**Weather false positive caught by the holdout.** On development data, Mumbai weather was the only
category in either market to pass the Holm-corrected rule (20d volatility). On the holdout, ridge with
weather (loss 0.1200) was not significantly better than random-walk volatility (p = 0.16), and plain HAR
on price features (0.1045) beat it. Conclusion unchanged: no auxiliary category is retained for NIFTY.
Full tables: `artifacts/nifty50/runs/.../holdout.csv`, `ablation_dev.csv`, and the UI with market = NIFTY 50.

## 9. Limitations

See [REMAINING_WORK.md](REMAINING_WORK.md) and [DATA_SOURCES.md](DATA_SOURCES.md). Most important:
single index; daily data only; free/unofficial price source; revised-vintage NFCI/Shiller data;
constant release lags; holdout is one ~6-year period; multiple-testing correction is per task/model
family, not experiment-wide. NIFTY 50 uses US macro data and lacks Indian macro, valuation and flow data
(see §8 and REMAINING_WORK.md).
