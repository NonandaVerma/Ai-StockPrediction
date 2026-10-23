# Project Specification: AI-Powered Multi-Factor Market Prediction System Using Python

## 1. Project Objective

Build a new Python-based market prediction system from scratch that identifies statistically meaningful patterns and forecasts future market behavior using historical data, financial indicators, macroeconomic conditions, company fundamentals, geopolitical events, weather, investor behavior, astronomical cycles, and other relevant information discovered through independent research.

The primary objective is to achieve the strongest possible **reproducible, out-of-sample predictive performance**, with robust risk estimates and realistic evaluation.

Do not assume that any category is predictive simply because it is included. Independently investigate which variables have credible economic mechanisms, sufficient historical coverage, and measurable predictive value.

Do not restrict the research to the categories listed in this prompt. Identify additional relevant categories where reliable evidence supports investigating them.

Build an entirely new, modular, maintainable Python application. Do not assume that any existing codebase, dataset, model, spreadsheet, database, API key, or infrastructure is available.

## 2. Research Before Implementation

Before choosing features, data providers, or model architectures:

1. Review established academic research on empirical asset pricing, return predictability, volatility forecasting, macroeconomic forecasting, event studies, financial machine learning, and portfolio risk.
2. Investigate the documented predictive value and limitations of candidate data categories.
3. Distinguish established findings, promising but uncertain hypotheses, and speculative ideas.
4. Identify publicly accessible, licensed, and commercially available data sources, including their historical coverage, publication frequency, revision policies, cost, and access limitations.
5. Prioritize features according to evidence, relevance to the target market, data quality, historical availability, and incremental predictive value.
6. Document the research methodology and cite genuine sources. Never invent papers, statistics, data availability, or results.

The system must remain open to discovering more useful features than those explicitly named below.

## 3. Market Coverage

Design the application to support configurable markets and asset classes:

- Indian equities: NSE and BSE.
- US equities: NYSE and NASDAQ.
- International equity indices.
- Agricultural commodities.
- Energy and metals.
- Foreign exchange.
- Government bonds and interest-rate markets.
- Futures and options, where reliable data is available.

Initially, choose one clearly defined market and prediction task for the first working version. Make other markets configurable extensions.

Do not combine markets with different trading calendars, liquidity characteristics, currency exposures, or reporting schedules without accounting for those differences.

## 4. Research and Collect Candidate Data

Develop independent data modules for the following categories. Determine their final priority through research and validation.

### A. Market prices, volume, and trading activity

Investigate:

- Historical open, high, low, close, adjusted close, and volume.
- Short-, medium-, and long-term returns.
- Momentum, trend, reversal, and relative strength.
- Realized volatility, downside volatility, and volatility clustering.
- Trading volume, turnover, liquidity, spreads, and order-book measures where available.
- Market breadth, sector participation, overnight gaps, and intraday ranges.
- Historical highs and lows, price jumps, and market stress indicators.

### B. Macroeconomics and monetary policy

Investigate:

- Inflation, GDP, industrial production, employment, wages, and retail sales.
- Central-bank policy rates and unexpected policy changes.
- Government bond yields, real interest rates, and yield-curve changes.
- Credit spreads, financial conditions, and funding stress.
- Purchasing managers' indices and business confidence.
- Economic forecasts, consensus estimates, and release surprises.
- International economic conditions and currency movements.

Use the historical values and publication timestamps available at each prediction date. Account for revisions to previously published data.

### C. Company fundamentals and corporate events

For individual securities, investigate:

- Revenue, earnings, cash flow, margins, debt, and financial strength.
- Valuation measures and their historical relationships with returns.
- Earnings surprises, guidance, and analyst estimate revisions.
- Dividends, buybacks, dilution, and corporate actions.
- Regulatory disclosures, management changes, acquisitions, and litigation.

Preserve historical company information and index membership where possible. Prevent survivorship bias.

### D. Cross-market and derivatives information

Investigate:

- Government bonds, currencies, commodities, and global indices.
- Volatility indices and implied volatility.
- Options-implied distributions and volatility skew.
- Futures curves, basis, open interest, and roll yield.
- Fund flows, short interest, institutional positioning, and market concentration.
- Relationships between sectors, industries, and their underlying economic drivers.

Only include features that are available at the prediction timestamp and appropriate for the asset being forecast.

### E. News, politics, and geopolitics

Investigate:

- Elections, policy changes, government budgets, and political uncertainty.
- Tariffs, sanctions, trade restrictions, and international conflicts.
- Central-bank announcements and economic policy decisions.
- Regulatory actions, corporate announcements, and industry developments.
- Supply-chain disruptions, shipping restrictions, and geopolitical shocks.
- News sentiment, uncertainty, novelty, relevance, and event severity.

Build an NLP pipeline to extract structured events from news where appropriate.

Store source, original publication time, time zone, affected entities, event type, novelty, and sentiment. Deduplicate syndicated reports and distinguish genuinely new information from repeated coverage.

Do not assume that positive news necessarily leads to positive returns: market expectations, surprise magnitude, valuation, and existing positioning can change the effect.

### F. Weather, climate, and natural disasters

Investigate:

- Rainfall, temperature anomalies, drought, floods, cyclones, and heatwaves.
- Soil moisture, crop conditions, and agricultural production estimates.
- Heating and cooling degree days.
- Wind, solar generation potential, and energy-demand forecasts.
- Weather-related infrastructure, shipping, transport, and supply-chain disruptions.
- Regional physical risks affecting commodity production and company operations.

Investigate NASA POWER, NOAA, and relevant national meteorological and agricultural agencies as potential sources.

Join observations and forecasts to geographically relevant assets and economic activities. For historical forecasting experiments, distinguish weather forecasts available at the time from weather actually observed later.

Prioritize weather variables where a credible economic transmission mechanism exists.

### G. Investor behavior and alternative data

Investigate:

- Investor sentiment surveys and positioning.
- Fund flows and institutional exposure.
- Short interest and insider transactions.
- Retail trading activity and investor attention.
- Search interest and publicly available behavioral indicators.
- Measures of crowded positioning and disagreement.

Assess licensing, reporting delays, coverage bias, and whether historical data can be reconstructed reliably.

### H. Astrology and astronomical cycles

Treat this as an explicitly experimental research hypothesis.

Investigate whether predefined astronomical variables provide incremental predictive information beyond standard financial, macroeconomic, seasonal, and event-related features.

Potential variables include:

- Planetary positions and angular relationships.
- Mercury retrograde periods.
- Lunar phases and dates relative to new or full moons.
- Eclipse dates.
- Other predefined astronomical cycles supported by a documented hypothesis.

Use a reliable ephemeris, such as Swiss Ephemeris, where appropriate. Document the calculation conventions and applicable licensing.

Run separate experiments for:

1. A standard financial model without astrology.
2. An astrology-only model.
3. A combined model.
4. Appropriate seasonal and calendar controls.

Use identical out-of-sample periods, correct for multiple testing, and measure incremental performance with uncertainty estimates.

Do not assume that these features predict markets. Reject them if their apparent benefits disappear under robust validation.

### I. Additional research-discovered categories

Identify other potentially useful features independently, including market microstructure, corporate supply-chain exposure, commodity inventories, options positioning, credit stress, economic forecast revisions, market concentration, and other documented predictive signals.

Include a new category only when the research identifies a meaningful hypothesis and a feasible way to test it.

## 5. Data Sources and Data Engineering

Build a provider-independent ingestion system.

Investigate appropriate sources such as:

- Official exchange and regulator data.
- Company filings and corporate disclosures.
- FRED and ALFRED for relevant US economic series and historical vintages.
- Central-bank and government statistical APIs.
- NASA POWER and NOAA for weather and climate data.
- CFTC positioning reports where relevant.
- Properly licensed market data, news archives, and alternative datasets.

For every source, document:

- Available fields and frequency.
- Historical depth and update schedule.
- Publication and revision timestamps.
- Licensing, cost, authentication, and rate limits.
- Missing-data characteristics and known limitations.

Implement retries, caching, rate limiting, schema validation, structured logging, and reproducible ingestion.

Do not assume paid APIs or credentials exist. Provide a working path using available data and clear configuration instructions for optional providers.

Use UTC internally for timestamps while preserving original time zones. Respect exchange calendars, market holidays, daylight-saving changes, and publication delays.

Store raw data separately from processed data and generated features. Use Parquet for analytical datasets and a suitable database for metadata or operational state when appropriate.

Never silently fabricate missing values or historical observations.

## 6. Define Prediction Targets

Create separate, configurable forecasting tasks.

### Target 1: Market direction

Estimate the probability that an asset's future return will be positive over 1, 5, and 20 trading days.

### Target 2: Expected return

Estimate the future percentage return over specified horizons.

### Target 3: Volatility

Forecast future realized volatility over appropriate horizons.

### Target 4: Downside risk

Estimate the probability of a loss exceeding specified thresholds and suitable tail-risk measures.

### Target 5: Market regime

Estimate probabilities of bull, bear, sideways, and high-volatility conditions using explicit definitions.

### Target 6: Event risk

Where data supports it, estimate the probability of significant price moves around scheduled or unexpected events.

Document the exact mathematical definition of every target, its forecast timestamp, holding period, and evaluation window.

Prevent overlapping targets from leaking information across validation boundaries.

## 7. Feature Engineering

Create reusable feature-engineering modules for every selected category.

Requirements:

- Generate features using information available at the prediction timestamp.
- Support multiple lookback windows and forecasting horizons.
- Preserve feature provenance and timestamps.
- Handle missingness, outliers, units, and corporate actions appropriately.
- Avoid redundant features and uncontrolled feature proliferation.
- Keep feature selection within the training pipeline.
- Measure category-level and individual-feature contributions.
- Investigate interactions only when supported by suitable validation.
- Monitor feature stability across different periods and market regimes.

Feature importance must not be interpreted as proof of causation.

## 8. Model Architecture

Start with reproducible, computationally practical models.

### Baseline models

- Logistic Regression.
- Linear Regression.
- Ridge and Elastic Net.
- Historical return and volatility benchmarks.

### Nonlinear machine learning

- LightGBM.
- XGBoost.
- CatBoost.
- Random Forest.

### Time-series models

- ARIMA where suitable.
- GARCH-family models for conditional volatility.
- Other time-series methods when justified by the target and data.

### Advanced models

- LSTM.
- Temporal convolutional networks.
- Transformers where data volume and validation justify their use.

Build separate models for different targets and horizons where necessary. Consider ensembles only when they demonstrate reliable out-of-sample improvement.

Do not assume that deep learning will outperform tree-based models or simpler statistical methods.

Use reproducible experiments, configuration-driven hyperparameter tuning, and documented model selection.

## 9. Training and Validation Requirements

Treat leakage prevention as a core system requirement.

Implement:

- Chronological training, validation, and final test periods.
- Expanding-window or rolling-window walk-forward validation.
- Purging or embargoes where overlapping labels require them.
- Training-only fitting of preprocessing, scaling, imputation, and feature selection.
- Point-in-time reconstruction of macroeconomic, news, fundamental, and weather data.
- Historical universe membership and delisted securities where relevant.
- An untouched final test period.
- Versioned datasets, feature definitions, model configurations, and results.

Do not use random train/test splits for ordinary financial time-series prediction.

Do not select a model using final test performance. Do not report training accuracy as evidence of predictive capability.

## 10. Evaluation and Model Selection

Use metrics appropriate to each task.

For direction:
- Balanced accuracy, precision, recall, F1, and ROC-AUC.

For probabilities:
- Brier score, log loss, and calibration.

For expected returns:
- MAE, RMSE, and out-of-sample R-squared.

For volatility and downside risk:
- Appropriate forecasting losses, tail-risk diagnostics, and threshold-based precision and recall.

For trading simulations:
- Net returns after transaction costs.
- Sharpe and Sortino ratios.
- Maximum drawdown.
- Turnover, liquidity, exposure, and slippage.
- Performance across different market regimes.

Compare every candidate model with suitable simple baselines.

Use confidence intervals and appropriate statistical tests. Account for repeated experimentation, multiple comparisons, and dependence between financial observations.

A model must not be considered successful solely because it achieves high directional accuracy. Assess calibration, robustness, economic value, and risk.

## 11. Category-Level Ablation Experiments

Create a repeatable experiment framework that compares:

1. Market prices and volume only.
2. Baseline plus macroeconomic data.
3. Baseline plus company fundamentals.
4. Baseline plus cross-market information.
5. Baseline plus news, politics, and geopolitical events.
6. Baseline plus weather and climate.
7. Baseline plus investor behavior and alternative data.
8. Baseline plus astrology and astronomical features.
9. Combined model using categories that survive validation.

Keep target definitions, test dates, evaluation methods, and comparable tuning budgets consistent.

For each experiment, report:

- Predictive performance relative to the baseline.
- Confidence intervals and statistical significance where appropriate.
- Performance across horizons and market regimes.
- Feature availability, missingness, and data costs.
- Sensitivity to model and feature choices.
- Whether the improvement persists after realistic trading costs, where applicable.

Do not retain a category merely because it improves training results.

## 12. Python Project Structure

Create a new, modular repository with a structure similar to:

- `src/data/` — ingestion, connectors, storage, validation.
- `src/research/` — literature review and data-source documentation.
- `src/features/` — feature engineering by category.
- `src/targets/` — target definitions and labeling.
- `src/models/` — statistical and machine-learning models.
- `src/training/` — training, tuning, and walk-forward validation.
- `src/evaluation/` — metrics, calibration, and risk analysis.
- `src/backtesting/` — optional realistic strategy simulations.
- `src/prediction/` — forecast generation and inference.
- `src/monitoring/` — model drift and performance tracking.
- `src/api/` — optional FastAPI service.
- `src/dashboard/` — optional Streamlit dashboard.
- `configs/` — validated settings for data, features, targets, and models.
- `tests/` — automated unit, integration, and leakage tests.
- `data/` — appropriately organized local datasets, excluded from version control when necessary.
- `artifacts/` — versioned model metadata and evaluation outputs.
- `docs/` — setup, research, data provenance, and methodology.

Use Python type hints, docstrings, structured logging, configuration management, reproducible dependencies, and automated tests.

Adapt the structure if a simpler design is more appropriate for the first version.

## 13. Forecast Output

For every prediction, generate a structured record containing:

- Asset identifier and market.
- Prediction timestamp.
- Forecast horizon and target.
- Probability of positive return.
- Expected return and uncertainty estimate, when supported.
- Predicted volatility.
- Downside-risk estimate.
- Market-regime probabilities.
- Model and dataset version.
- Feature coverage and data-quality warnings.
- Applicable validation and calibration metadata.

Do not label an uncalibrated score as a reliable probability. Distinguish forecasts from actual observed outcomes and historical backtests.

## 14. Dashboard and API

After the core training and evaluation pipeline works, create an optional dashboard using Streamlit and an API using FastAPI.

Display:

- Forecasts by asset and horizon.
- Historical forecasts versus actual outcomes.
- Model and baseline comparisons.
- Feature-category ablation results.
- Risk and market-regime estimates.
- Data freshness and quality warnings.
- Model drift and performance monitoring.
- Historical backtest versus paper-trading results.

Keep dashboard functionality separate from the underlying research and training code.

## 15. Development and Testing

Implement the system in incremental, runnable stages.

### Stage 1: Research and planning
Produce the research report, source evaluation, initial market choice, target definitions, and implementation plan.

### Stage 2: Project foundation
Create the Python repository, environment setup, configuration, logging, tests, and documentation.

### Stage 3: Data pipeline
Implement reliable data ingestion, storage, validation, timestamp alignment, and point-in-time reconstruction.

### Stage 4: Initial dataset and baseline
Build the first usable dataset, define targets, and train simple benchmark models.

### Stage 5: Feature engineering
Implement selected categories with tests for correctness, leakage, and historical availability.

### Stage 6: Model experiments
Compare statistical models, tree-based models, and other justified candidates using walk-forward validation.

### Stage 7: Feature ablation
Test incremental category contributions, including weather, political events, and astrology, without presuming their usefulness.

### Stage 8: Evaluation and risk
Produce final out-of-sample reports and realistic cost-aware simulations where appropriate.

### Stage 9: Monitoring and interface
Add inference, model versioning, drift monitoring, and optional dashboard/API functionality.

### Stage 10: Final documentation
Provide exact setup and execution commands, test results, known limitations, and reproducibility instructions.

## 16. Final Requirements

- Build a completely new project without relying on previous conversations, existing spreadsheets, or unprovided files.
- Do not fabricate historical data, research results, performance figures, or API capabilities.
- Do not hard-code credentials or assume paid services are available.
- Do not introduce unnecessary complexity before the baseline works.
- Do not use future information during training or evaluation.
- Do not promise a particular accuracy percentage or guaranteed profits.
- Make all feature categories configurable and independently testable.
- Preserve reproducibility through versioned datasets, configurations, and experiments.
- Explain any implementation that cannot be completed due to missing data, credentials, licensing, or compute resources.

**Start by researching the problem and creating the project plan. Then implement the complete working system in stages, running tests and documenting actual results at each stage. Optimize for demonstrable out-of-sample predictive value, not complexity or the number of features.**