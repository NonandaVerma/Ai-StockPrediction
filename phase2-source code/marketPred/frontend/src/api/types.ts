// Typed response interfaces for the marketpred FastAPI backend (see docs/API.md).

export type TaskKind = "binary" | "regression" | "volatility" | "multiclass";

export interface TaskSpec {
  task: string;
  kind: TaskKind;
  target: string;
  horizon: number;
}

export interface RunInfo {
  name: string;
  created_utc: string;
  dataset_version: string;
  config_fingerprint: string;
  feature_sets: Record<string, string[]>;
  stages_complete: { comparison: boolean; ablation: boolean; selection: boolean; holdout: boolean };
  holdout_evaluations: { utc: string; run_dir: string; dataset_version: string; config: string }[];
  validation: {
    dev_start: string;
    holdout_start: string;
    retrain_every_days: number;
    embargo_days: number;
    min_train_obs: number;
    calibration_fraction: number;
  };
}

export interface Market {
  id: string;
  name: string;
  ticker: string;
  calendar: string;
  timezone: string;
  currency: string;
  price_coverage: { start: string; end: string };
  tasks: TaskSpec[];
  run: RunInfo | null;
}

export interface PriceRow {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number;
  volume: number | null;
  return_1d: number | null;
}

export interface PricesResponse {
  market: string;
  kind: string;
  open_reliable_from: string | null;
  rows: PriceRow[];
}

export interface AvailablePrediction {
  period: "dev" | "holdout";
  task: string;
  model: string;
  feature_set: string;
  kind: TaskKind;
  horizon: number;
  selected: boolean;
  reference: boolean;
}

/** One stored walk-forward prediction. Fields present depend on the task kind. */
export interface PredictionRow {
  prediction_date: string;
  target_date: string;
  close_at_prediction: number | null;
  actual_close_target: number | null;
  actual_return: number | null;
  fold: number;
  train_end: string;
  // regression (returns)
  predicted_return?: number;
  predicted_price?: number;
  price_error?: number;
  price_error_pct?: number;
  return_error?: number;
  // volatility
  predicted_vol?: number;
  actual_vol?: number;
  vol_error?: number;
  vol_error_pct?: number;
  // binary
  probability?: number;
  actual_event?: number;
  predicted_event?: number;
  prob_error?: number;
  // direction (binary or regression)
  predicted_direction?: "up" | "down";
  actual_direction?: "up" | "down";
  // regime
  actual_regime?: string;
  predicted_regime?: string;
  [key: string]: unknown;
}

export interface PredictionsMeta {
  market: string;
  task: string;
  kind: TaskKind;
  horizon: number;
  target: string;
  model: string;
  feature_set: string;
  period: string;
  method: string;
  leakage_free: boolean;
  retrospective_fit: boolean;
  n_folds: number;
  n: number;
  calibrated: boolean;
  price_conversion: string | null;
}

export interface PredictionsResponse {
  meta: PredictionsMeta;
  rows: PredictionRow[];
}

export type MetricMap = Record<string, number>;

export interface PairedTest {
  n_common: number;
  loss_ref: number;
  loss_model: number;
  loss_improvement: number;
  rel_improvement: number;
  dm_stat: number;
  p_value: number;
  ci_low: number;
  ci_high: number;
  auc_diff?: number;
  auc_diff_ci_low?: number;
  auc_diff_ci_high?: number;
}

export interface EvaluationResponse {
  n: number;
  start: string;
  end: string;
  metrics: MetricMap;
  price_metrics?: { mae: number; rmse: number; mape_pct: number };
  baseline: { model: string; metrics: MetricMap; comparison: PairedTest } | null;
  by_vol_regime: Record<string, MetricMap>;
  by_vol_regime_improvement_vs_baseline?: Record<string, number>;
  vol_regime_definition: string;
}

export interface FeatureInfo {
  feature: string;
  category: string;
  first_valid: string | null;
  last_valid: string | null;
  coverage: number;
  source?: string;
  transform?: string;
}

export interface SourceStatus {
  fetched_at: string;
  source: string;
  dataset: string;
  status: string;
  rows: number;
  min_date: string;
  max_date: string;
  message: string;
}

export interface StatusResponse {
  market: string;
  sources: SourceStatus[];
  price_coverage: { start: string; end: string; rows: number };
  feature_dataset: { dataset_version: string; rows: number; first_date: string; last_date: string };
  stale_features: { feature: string; category: string; last_valid: string; days_stale: number }[];
  run: RunInfo | null;
  forecast_available: boolean;
}

export interface HorizonForecast {
  prob_positive_return: number | null;
  prob_calibration: string;
  expected_log_return: number | null;
  return_uncertainty_sd: number | null;
  return_interval_90: [number, number] | null;
  predicted_volatility_annualized: number | null;
  volatility_source_task: string | null;
  value_at_risk_95: number | null;
  prob_loss_beyond_threshold: number | null;
  loss_threshold: number | null;
}

export interface Forecast {
  record_type: "forecast";
  asset: string;
  market: string;
  market_name: string;
  prediction_timestamp_utc: string;
  data_as_of_session: string;
  generated_at_utc: string;
  horizons_trading_days: Record<string, HorizonForecast>;
  large_move_probability_1d: number | null;
  regime_horizon_days: number;
  regime_probabilities: Record<string, number> | null;
  regime_definitions: Record<string, number>;
  versions: Record<string, string>;
  feature_coverage: Record<string, number>;
  data_quality_warnings: string[];
  tasks: Record<string, { model: string; feature_set: string; categories: string[]; calibrated: boolean; train_end: string; n_train: number }>;
  disclaimer: string;
}

/** Row of ablation_dev.csv (subset of columns used by the UI). */
export interface AblationRow {
  task: string;
  kind: TaskKind;
  horizon: number;
  model: string;
  feature_set: string;
  n_features: number;
  test_coverage_added: number;
  loss_improvement: number | null;
  rel_improvement: number | null;
  ci_low: number | null;
  ci_high: number | null;
  p_value: number | null;
  holm_p: number | null;
  significant_improvement: boolean;
  [key: string]: unknown;
}

export type ComparisonRow = Record<string, string | number | boolean | null>;

export interface Selection {
  created_utc: string;
  rule: string;
  tasks: Record<string, {
    kind: TaskKind;
    primary_ablation_model: string;
    surviving_categories: string[];
    combined_feature_set: string[];
    best_dev_model: string;
    best_dev_feature_set: string;
    best_dev_loss: number;
    loss_metric: string;
    reference: string;
  }>;
}

export interface TradingResult {
  strategy: Record<string, number>;
  buy_and_hold: Record<string, number>;
  exposure: number;
  turnover_per_year: number;
  n_trades: number;
  cost_bps_round_trip: number;
  by_vol_regime: Record<string, { strategy: Record<string, number>; buy_and_hold: Record<string, number>; exposure: number }>;
}
