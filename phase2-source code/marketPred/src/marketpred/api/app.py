"""Read-only FastAPI service over generated artifacts (no training or model fitting in request handlers).

Run: uvicorn marketpred.api.app:app --port 8000
UI contract: every `/api/*` route below is documented in docs/API.md.
"""

from __future__ import annotations

import json
from functools import wraps
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ..config import load_config
from . import service

app = FastAPI(title="marketpred", version="0.2.0",
              description="Research forecasts, stored walk-forward predictions and validation results. "
                          "Not investment advice.")
app.add_middleware(CORSMiddleware, allow_origins=service.cors_origins(), allow_methods=["GET"],
                   allow_headers=["*"])


def handled(fn):
    """Map data-availability problems to clean HTTP errors."""
    @wraps(fn)
    def inner(*a, **kw):
        try:
            return fn(*a, **kw)
        except HTTPException:
            raise
        except (FileNotFoundError, KeyError) as e:
            raise HTTPException(404, str(e).strip("'\""))
        except ValueError as e:
            raise HTTPException(422, str(e))
    return inner


# ---- legacy routes (kept for compatibility) --------------------------------------------------------
def _cfg():
    return load_config()


def _latest_run_dir() -> Path:
    runs = sorted((_cfg().artifacts_dir / "runs").glob("*/manifest.json"))
    if not runs:
        raise HTTPException(404, "no experiment runs found")
    return runs[-1].parent


def _csv(path: Path) -> list[dict]:
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found")
    return json.loads(pd.read_csv(path).to_json(orient="records"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/forecast/latest")
def forecast_latest() -> dict:
    cfg = _cfg()
    path = cfg.artifacts_dir / "predictions" / f"latest_{cfg.market.id}.json"
    if not path.exists():
        raise HTTPException(404, "no forecast yet; run `marketpred predict`")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/results/{name}")
def results(name: str) -> list[dict]:
    allowed = {"comparison_dev", "ablation_dev", "holdout", "var_backtest_dev", "var_backtest_holdout"}
    if name not in allowed:
        raise HTTPException(404, f"unknown result; choose from {sorted(allowed)}")
    return _csv(_latest_run_dir() / f"{name}.csv")


@app.get("/selection")
def selection() -> dict:
    path = _latest_run_dir() / "selection.json"
    if not path.exists():
        raise HTTPException(404, "selection not available")
    return json.loads(path.read_text(encoding="utf-8"))


# ---- UI API -------------------------------------------------------------------------------------------
@app.get("/api/health")
def api_health() -> dict:
    return {"status": "ok", "version": app.version}


@app.get("/api/markets")
@handled
def api_markets() -> list[dict]:
    out = []
    for mid in service.market_configs():
        cfg = service.get_cfg(mid)
        px = service.prices(mid)
        tasks = service.task_table(cfg)
        out.append({"id": mid, "name": cfg.market.name, "ticker": cfg.market.ticker, "calendar": cfg.market.calendar,
                    "timezone": cfg.market.timezone, "currency": cfg.market.currency,
                    "price_coverage": {"start": str(px.index.min().date()), "end": str(px.index.max().date())},
                    "tasks": list(tasks.values()), "run": service.run_info(mid)})
    return out


@app.get("/api/prices")
@handled
def api_prices(market: str = "SPX", start: str | None = None, end: str | None = None) -> dict:
    px = service.prices(market)
    T = service.targets(market)
    df = px.copy()
    df["return_1d"] = T["fwd_ret_1"].shift(1).reindex(df.index)  # realised log return into each date
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    df = df.drop(columns=["adj_close"], errors="ignore")
    df.insert(0, "date", df.index.strftime("%Y-%m-%d"))
    reliable_from = "2007-01-01" if market == "SPX" else None
    return {"market": market, "kind": "observed end-of-day data (delayed, not live)",
            "open_reliable_from": reliable_from,
            "rows": json.loads(df.to_json(orient="records"))}


@app.get("/api/predictions/available")
@handled
def api_available(market: str = "SPX") -> list[dict]:
    return service.list_predictions(market)


@app.get("/api/predictions")
@handled
def api_predictions(market: str = "SPX", task: str = Query(...), model: str = Query(...),
                    feature_set: str = "price", period: str = "dev", start: str | None = None,
                    end: str | None = None) -> dict:
    return service.prediction_rows(market, period, task, model, feature_set, start, end)


@app.get("/api/evaluation")
@handled
def api_evaluation(market: str = "SPX", task: str = Query(...), model: str = Query(...),
                   feature_set: str = "price", period: str = "dev", start: str | None = None,
                   end: str | None = None) -> dict:
    return service.evaluate(market, period, task, model, feature_set, start, end)


@app.get("/api/run/{name}")
@handled
def api_run_table(name: str, market: str = "SPX"):
    allowed = {"comparison_dev": "comparison_dev.csv", "ablation_dev": "ablation_dev.csv",
               "holdout": "holdout.csv", "selection": "selection.json", "trading_dev": "trading_dev.json",
               "trading_holdout": "trading_holdout.json", "var_backtest_dev": "var_backtest_dev.csv",
               "var_backtest_holdout": "var_backtest_holdout.csv"}
    if name not in allowed:
        raise HTTPException(404, f"unknown table; choose from {sorted(allowed)}")
    return service.run_table(market, allowed[name])


@app.get("/api/features")
@handled
def api_features(market: str = "SPX") -> list[dict]:
    return service.feature_catalog(market)


@app.get("/api/forecast/latest")
@handled
def api_forecast(market: str = "SPX") -> dict:
    f = service.latest_forecast(market)
    if f is None:
        raise HTTPException(404, "no forecast generated yet; run `marketpred predict`")
    return f


@app.get("/api/status")
@handled
def api_status(market: str = "SPX") -> dict:
    return service.status(market)
