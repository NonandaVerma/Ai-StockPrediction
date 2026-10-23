// Centralised API client. The backend base URL comes from VITE_API_BASE_URL.
import type {
  AblationRow, AvailablePrediction, ComparisonRow, EvaluationResponse, FeatureInfo, Forecast, Market,
  PredictionsResponse, PricesResponse, Selection, StatusResponse, TradingResult,
} from "./types";

export const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

type Params = Record<string, string | number | undefined | null>;

async function get<T>(path: string, params: Params = {}): Promise<T> {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  const url = `${API_BASE}${path}${qs.toString() ? `?${qs}` : ""}`;
  let res: Response;
  try {
    res = await fetch(url);
  } catch {
    throw new ApiError(0, `Cannot reach the backend at ${API_BASE}. Start it with: uvicorn marketpred.api.app:app --port 8000`);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch { /* non-JSON error body */ }
    throw new ApiError(res.status, String(detail));
  }
  return res.json() as Promise<T>;
}

export interface PredictionQuery {
  market: string;
  task: string;
  model: string;
  feature_set: string;
  period: string;
  start?: string;
  end?: string;
}

export const api = {
  health: () => get<{ status: string; version: string }>("/api/health"),
  markets: () => get<Market[]>("/api/markets"),
  prices: (market: string, start?: string, end?: string) => get<PricesResponse>("/api/prices", { market, start, end }),
  available: (market: string) => get<AvailablePrediction[]>("/api/predictions/available", { market }),
  predictions: (q: PredictionQuery) => get<PredictionsResponse>("/api/predictions", { ...q }),
  evaluation: (q: PredictionQuery) => get<EvaluationResponse>("/api/evaluation", { ...q }),
  ablation: (market: string) => get<AblationRow[]>("/api/run/ablation_dev", { market }),
  comparison: (market: string) => get<ComparisonRow[]>("/api/run/comparison_dev", { market }),
  holdout: (market: string) => get<ComparisonRow[]>("/api/run/holdout", { market }),
  selection: (market: string) => get<Selection>("/api/run/selection", { market }),
  trading: (market: string, period: "dev" | "holdout") =>
    get<Record<string, TradingResult>>(`/api/run/trading_${period}`, { market }),
  varBacktest: (market: string, period: "dev" | "holdout") =>
    get<ComparisonRow[]>(`/api/run/var_backtest_${period}`, { market }),
  features: (market: string) => get<FeatureInfo[]>("/api/features", { market }),
  forecast: (market: string) => get<Forecast>("/api/forecast/latest", { market }),
  status: (market: string) => get<StatusResponse>("/api/status", { market }),
};
