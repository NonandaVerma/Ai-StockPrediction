# Research review and feature prioritisation

This document records the research basis for the first version (S&P 500 index, daily decisions,
1/5/20-session horizons). It separates **established findings**, **promising but uncertain
hypotheses** and **speculative ideas**, and explains which hypotheses were implemented and how they are
tested. Citations are to published work; where I could not identify supporting peer-reviewed evidence,
that is stated explicitly rather than filled in. Results from *this* project are in `docs/REPORT.md` and
`docs/RESULTS.md`; nothing here is a result of this project.

## 1. Methodology of the review

1. Start from survey-level and benchmark papers on return predictability, volatility forecasting and
   financial machine learning.
2. For each candidate data category, ask: (a) is there a credible economic mechanism, (b) is there
   out-of-sample evidence, (c) can the data be reconstructed **point-in-time** for 20+ years from sources
   we can legally and actually access, (d) what is the incremental value over price-based information.
3. Fix the feature definitions *before* running ablations (pre-specification), keep the number of
   features per category small, and test every category against the same baseline with multiple-testing
   correction.

## 2. Core evidence that shapes the design

### Established

* **Equity-premium predictability is weak out of sample.** Welch & Goyal (2008, *Review of Financial
  Studies* 21(4)) show that most classic predictors (valuation ratios, rates, spreads) fail to beat the
  historical mean out of sample. Campbell & Thompson (2008, *RFS* 21(4)) find small positive
  out-of-sample R² once economically motivated restrictions are imposed, and argue that even tiny R²
  can matter economically. Rapach, Strauss & Zhou (2010, *RFS* 23(2)) find forecast combination helps.
  **Implication:** the historical mean is the benchmark to beat for returns; expected R² is ~0-1%.
* **Volatility is highly predictable.** Volatility clustering (Engle 1982, *Econometrica* 50(4);
  Bollerslev 1986, *Journal of Econometrics* 31(3)); realised-volatility models (Andersen, Bollerslev,
  Diebold & Labys 2003, *Econometrica* 71(2)); the HAR model (Corsi 2009, *Journal of Financial
  Econometrics* 7(2)). **Implication:** volatility, downside-risk and regime targets should be far
  more forecastable than direction; HAR and GARCH are the baselines.
* **Time-series momentum** across asset classes (Moskowitz, Ooi & Pedersen 2012, *Journal of Financial
  Economics* 104(2)); cross-sectional momentum (Jegadeesh & Titman 1993, *Journal of Finance* 48(1)).
  Evidence is strongest at monthly horizons and for diversified portfolios, weaker for a single index
  at daily horizons.
* **Machine learning** can improve return prediction in large cross-sections (Gu, Kelly & Xiu 2020,
  *RFS* 33(5)), with gains concentrated in nonlinear interactions among many stock-level signals. A
  single index time series has ~9,000 daily observations and a low signal-to-noise ratio, so deep
  learning is not justified for v1.
* **Backtest overfitting and multiple testing** inflate apparent skill (Sullivan, Timmermann & White
  1999, *Journal of Finance* 54(5); Bailey, Borwein, López de Prado & Zhu 2014, *Notices of the AMS*
  61(5); Harvey, Liu & Zhu 2016, *RFS* 29(1)). **Implication:** fixed protocols, purged walk-forward
  validation (López de Prado 2018, *Advances in Financial Machine Learning*, Wiley), Holm (1979,
  *Scandinavian Journal of Statistics* 6(2)) correction and an untouched holdout.
* **Forecast comparison** with serially dependent losses: Diebold & Mariano (1995, *Journal of Business
  & Economic Statistics* 13(3)) with Newey & West (1987, *Econometrica* 55(3)) HAC variance; block
  bootstrap (Politis & Romano 1994, *JASA* 89(428), stationary bootstrap; we use the circular block
  variant). Volatility losses robust to proxy noise: QLIKE and MSE (Patton 2011, *Journal of
  Econometrics* 160(1)). VaR coverage: Kupiec (1995, *Journal of Derivatives* 3(2)).
* **Real-time macro data differ from revised data** (Croushore & Stark 2001, *Journal of Econometrics*
  105(1)). Using today's vintage of revised series in a backtest leaks information.

### Category-level assessment

| Category | Mechanism | Evidence | Status in v1 |
|---|---|---|---|
| A. Prices/volume (momentum, reversal, realised vol, ranges) | Under-reaction, risk premia, vol clustering | Established for volatility; weak for daily index direction | Baseline category |
| B. Macro & rates (term spread, short rate, financial conditions, CPI, unemployment, sentiment) | Business cycle / discount rates; term spread predicts recessions (Estrella & Mishkin 1998, *REStat* 80(1)); term/default spreads track expected returns (Fama & French 1989, *JFE* 25(1)) | Mostly in-sample; weak out of sample (Welch & Goyal 2008) | Implemented, tested |
| C. Fundamentals (index valuation: E/P, D/P, CAPE-style) | Present-value identity (Campbell & Shiller 1988, *RFS* 1(3)); Fama & French (1988, *JFE* 22(1)) | Long-horizon in-sample evidence; weak OOS, slow-moving | Implemented, tested (limited by data ending 2024) |
| D. Cross-asset & derivatives (VIX, term structure, variance risk premium, credit, dollar, commodities, size spread) | Risk appetite; variance risk premium predicts returns at multi-month horizons (Bollerslev, Tauchen & Zhou 2009, *RFS* 22(11)); credit spreads (Gilchrist & Zakrajšek 2012, *AER* 102(4)) | Strong for volatility, moderate for returns | Implemented, tested |
| E. News/politics/geopolitics | Uncertainty shocks; media pessimism predicts short-term reversal (Tetlock 2007, *JF* 62(3)); EPU (Baker, Bloom & Davis 2016, *QJE* 131(4)); GPR (Caldara & Iacoviello 2022, *AER* 112(4)) | Uncertainty indices relate to volatility and investment; short-horizon return evidence mixed | Implemented via published daily indices; article-level NLP pipeline implemented but untested on real archive (no licensed data) |
| F. Weather | Mood: NYC sunshine and returns (Saunders 1993, *AER* 83(5); Hirshleifer & Shumway 2003, *JF* 58(3)); seasonal affective disorder (Kamstra, Kramer & Levi 2003, *AER* 93(1)) | Small, contested, weakened since electronic trading dispersed traders | Implemented as an **experimental** hypothesis for the index; a much stronger mechanism exists for agricultural and energy markets (future extension) |
| G. Investor behaviour & positioning | Sentiment and limits to arbitrage (Baker & Wurgler 2006, *JF* 61(4)); attention via search (Da, Engelberg & Gao 2011, *JF* 66(5)); CFTC positioning (mixed; e.g. Sanders, Boris & Manfredo 2004, *Energy Economics* 26(3)) | Sentiment indices are monthly and partly revised; daily evidence weak | CFTC COT implemented; AAII blocked; Google Trends not reproducible point-in-time (sampled, re-normalised) |
| H. Astrology/astronomy | No accepted economic mechanism. Lunar-cycle return differences reported by Dichev & Janes (2003, *Journal of Private Equity* 6(4)) and Yuan, Zheng & Zhu (2006, *Journal of Empirical Finance* 13(1)), attributed to mood | Small effects in some samples; I found no peer-reviewed evidence for Mercury retrograde, eclipses or planetary aspects | Implemented as a **pre-specified speculative** hypothesis with the required four-way protocol |
| I. Calendar/seasonality (added) | Settlement/flow effects; turn-of-month (Ariel 1987, *JFE* 18(1)); holiday and turn-of-month across 90 years (Lakonishok & Smidt 1988, *RFS* 1(4)); Halloween effect (Bouman & Jacobsen 2002, *AER* 92(5)) | Documented but decaying; must be the control for astrology | Implemented (control category) |

### Promising but not implementable here (and why)

* **Options-implied distributions and skew** (e.g. SKEW index, risk-neutral moments): CBOE SKEW is
  downloadable but the free history and definitions changed; full option surfaces require OptionMetrics
  or exchange data (paid).
* **Order book, spreads, intraday realised variance**: requires TAQ/LOBSTER-style data (paid). With
  intraday data, HAR on 5-minute RV would replace the daily-range proxy.
* **Earnings surprises, estimate revisions, short interest, insider trades, fund flows** for single
  stocks: require point-in-time vendor databases (I/B/E/S, Compustat Snapshot, FINRA files with history,
  EPFR). Not used because survivorship-free, point-in-time reconstruction is not possible from free sources.
* **Macro release surprises** (actual minus consensus): consensus history is proprietary (Bloomberg/Refinitiv).
* **Weather forecasts as of the decision time**: historical forecast archives (e.g. ECMWF/TIGGE, NOAA
  GEFS reforecasts) exist but are large and only relevant to weather-sensitive markets.

## 3. Initial market and task choice

**S&P 500 price index (`^GSPC`), daily decisions at the NYSE close.** Reasons: longest freely
available daily history, the richest set of point-in-time-reconstructable US auxiliary data
(Treasury, BLS, Chicago Fed, CBOE, CFTC, EPU/GPR), a single trading calendar and currency, and no
survivorship issue at the index level. Indian equities (NIFTY 50, `configs/nifty50.yaml`) are configured
as an extension; mixing markets is avoided because calendars, currencies and data timing differ.

## 4. Pre-specified expectations (written before running experiments)

1. Direction at 1/5/20 days: little or no out-of-sample skill; calibrated probabilities close to the
   base rate. Any apparent edge must survive costs and multiple testing.
2. Volatility and downside/large-move probabilities: clear skill relative to unconditional baselines,
   driven mainly by price-based volatility and VIX.
3. Astrology, weather and most macro categories: no incremental value at daily horizons.

These are hypotheses, not results; see `docs/REPORT.md` for what the data showed.
