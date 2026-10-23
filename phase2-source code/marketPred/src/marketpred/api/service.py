"""Read-only data access for the UI: prices, stored walk-forward predictions, and evaluation.

Nothing here trains models or generates predictions. Historical predictions are the out-of-sample
records saved by the walk-forward engine (`training/walkforward.py`): each was produced by a model
fitted only on purged data available before its prediction date. Metrics reuse the project's
evaluation code so the UI and the reports agree.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT, Config, load_config
from ..data.store import DataStore
from ..evaluation.metrics import summarize
from ..targets.labels import REGIMES, build_targets, task_specs

VOL_BINS, VOL_LABELS = [0, 0.12, 0.20, np.inf], ["low_vol", "mid_vol", "high_vol"]
METHOD = ("walk-forward out-of-sample: expanding training window refit every N sessions, labels purged "
          "(horizon + embargo) before each test block; preprocessing, tuning and calibration fitted "
          "on training data only")


# ---- markets and runs -----------------------------------------------------------------------------
def market_configs() -> dict[str, Path | None]:
    """Market id -> config override path (None = default). Only markets with built features are listed."""
    out: dict[str, Path | None] = {}
    for path in [None, *sorted((PROJECT_ROOT / "configs").glob("*.yaml"))]:
        if path is not None and path.name == "default.yaml":
            continue
        cfg = load_config(path)
        if (cfg.data_dir / "features" / f"{cfg.market.id}_features.parquet").exists():
            out.setdefault(cfg.market.id, path)
    return out


def get_cfg(market: str) -> Config:
    m = market_configs()
    if market not in m:
        raise KeyError(f"unknown market {market!r}; available: {sorted(m)}")
    return load_config(m[market])


def latest_run(cfg: Config) -> Path | None:
    runs = sorted((cfg.artifacts_dir / "runs").glob("*/manifest.json"))
    return runs[-1].parent if runs else None


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


# ---- cached data ------------------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _prices(market: str, mtime: float) -> pd.DataFrame:
    cfg = get_cfg(market)
    return DataStore(cfg.data_dir).load_prices(cfg.market.ticker)


def prices(market: str) -> pd.DataFrame:
    cfg = get_cfg(market)
    path = cfg.data_dir / "processed" / "prices"
    return _prices(market, max((_mtime(p) for p in path.glob("*.parquet")), default=0))


@lru_cache(maxsize=8)
def _targets(market: str, mtime: float) -> pd.DataFrame:
    return build_targets(prices(market)["close"], get_cfg(market).targets)


def targets(market: str) -> pd.DataFrame:
    return _targets(market, prices(market).index[-1].value)


@lru_cache(maxsize=8)
def _features(market: str, mtime: float) -> tuple[pd.DataFrame, dict]:
    cfg = get_cfg(market)
    p = cfg.data_dir / "features" / f"{cfg.market.id}_features.parquet"
    return pd.read_parquet(p), json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))


def features(market: str) -> tuple[pd.DataFrame, dict]:
    cfg = get_cfg(market)
    return _features(market, _mtime(cfg.data_dir / "features" / f"{cfg.market.id}_features.parquet"))


def vol_regimes(market: str) -> pd.Series:
    X, _ = features(market)
    return pd.cut(X["price__rv_63"], VOL_BINS, labels=VOL_LABELS).astype(str)


# ---- prediction inventory -------------------------------------------------------------------------
def task_table(cfg: Config) -> dict[str, dict]:
    return {t["task"]: t for t in task_specs(cfg.targets)}


def list_predictions(market: str) -> list[dict]:
    cfg = get_cfg(market)
    run = latest_run(cfg)
    if run is None:
        return []
    sel_path = run / "selection.json"
    sel = json.loads(sel_path.read_text(encoding="utf-8"))["tasks"] if sel_path.exists() else {}
    tasks = task_table(cfg)
    rows = []
    for f in sorted((run / "preds").glob("*.parquet")):
        period, task, model, fs = f.stem.split("__")
        fs = fs.replace("-", "+")
        if task not in tasks:
            continue
        s = sel.get(task, {})
        rows.append({"period": period, "task": task, "model": model, "feature_set": fs,
                     "kind": tasks[task]["kind"], "horizon": tasks[task]["horizon"],
                     "selected": s.get("best_dev_model") == model and s.get("best_dev_feature_set") == fs,
                     "reference": model in ("climatology", "hist_mean", "rw_vol")})
    return rows


def _pred_path(market: str, period: str, task: str, model: str, fs: str) -> Path:
    run = latest_run(get_cfg(market))
    if run is None:
        raise FileNotFoundError("no experiment run")
    p = run / "preds" / f"{period}__{task}__{model}__{fs.replace('+', '-')}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"no stored predictions for {period}/{task}/{model}/{fs}")
    return p


def load_predictions(market: str, period: str, task: str, model: str, fs: str,
                     start: str | None = None, end: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(_pred_path(market, period, task, model, fs))
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df


# ---- row construction -----------------------------------------------------------------------------
def prediction_rows(market: str, period: str, task: str, model: str, fs: str,
                    start: str | None = None, end: str | None = None) -> dict:
    cfg = get_cfg(market)
    spec = task_table(cfg)[task]
    kind, h = spec["kind"], spec["horizon"]
    df = load_predictions(market, period, task, model, fs, start, end)
    px, T = prices(market)["close"], targets(market)
    idx = df.index
    out = pd.DataFrame(index=idx)
    out["prediction_date"] = idx.strftime("%Y-%m-%d")
    target_date = T[f"label_end_{h}"].reindex(idx)
    out["target_date"] = pd.to_datetime(target_date).dt.strftime("%Y-%m-%d")
    out["close_at_prediction"] = px.reindex(idx).to_numpy()
    out["actual_close_target"] = px.reindex(pd.DatetimeIndex(target_date)).to_numpy()
    out["actual_return"] = T[f"fwd_ret_{h}"].reindex(idx).to_numpy()
    out["fold"] = df["fold"].to_numpy()
    out["train_end"] = pd.to_datetime(df["train_end"]).dt.strftime("%Y-%m-%d").to_numpy()
    if kind == "multiclass":
        P = df[[f"p{i}" for i in range(len(REGIMES))]].to_numpy()
        for i, r in enumerate(REGIMES):
            out[f"prob_{r}"] = P[:, i]
        out["actual_regime"] = [REGIMES[int(v)] for v in df["y"]]
        out["predicted_regime"] = [REGIMES[i] for i in P.argmax(axis=1)]
    elif kind == "regression":  # log-return forecasts -> comparable price path
        out["predicted_return"] = df["pred"].to_numpy()
        out["predicted_price"] = out["close_at_prediction"] * np.exp(out["predicted_return"])
        out["predicted_direction"] = np.where(out["predicted_return"] > 0, "up", "down")
        out["actual_direction"] = np.where(out["actual_return"] > 0, "up", "down")
        out["price_error"] = out["actual_close_target"] - out["predicted_price"]
        out["price_error_pct"] = out["price_error"] / out["actual_close_target"] * 100
        out["return_error"] = out["actual_return"] - out["predicted_return"]
    elif kind == "volatility":
        out["predicted_vol"] = np.exp(df["pred"].to_numpy())
        out["actual_vol"] = np.exp(df["y"].to_numpy())
        out["vol_error"] = out["actual_vol"] - out["predicted_vol"]
        out["vol_error_pct"] = out["vol_error"] / out["actual_vol"] * 100
    else:  # binary: direction / downside / large move
        out["probability"] = df["pred"].to_numpy()
        out["actual_event"] = df["y"].astype(int).to_numpy()
        out["predicted_event"] = (df["pred"] > 0.5).astype(int).to_numpy()
        if task.startswith("direction"):
            out["predicted_direction"] = np.where(out["probability"] > 0.5, "up", "down")
            out["actual_direction"] = np.where(out["actual_event"] == 1, "up", "down")
        out["prob_error"] = out["actual_event"] - out["probability"]
    out = out.replace([np.inf, -np.inf], np.nan)
    meta = {"market": market, "task": task, "kind": kind, "horizon": h, "target": spec["target"], "model": model,
            "feature_set": fs, "period": period, "method": METHOD, "leakage_free": True,
            "retrospective_fit": False, "n_folds": int(df["fold"].nunique()), "n": int(len(df)),
            "calibrated": bool(kind == "binary" and model in cfg.models.calibrate),
            "price_conversion": ("predicted_price = close_at_prediction * exp(predicted log return over h sessions); "
                                 "uses only the close known at the prediction date") if kind == "regression" else None}
    return {"meta": meta, "rows": json.loads(out.to_json(orient="records"))}


# ---- evaluation -----------------------------------------------------------------------------------
def _reference(kind: str) -> str:
    return {"binary": "climatology", "multiclass": "climatology", "regression": "hist_mean",
            "volatility": "rw_vol"}[kind]


def evaluate(market: str, period: str, task: str, model: str, fs: str,
             start: str | None = None, end: str | None = None) -> dict:
    from ..training.experiments import paired_test, regime_breakdown, score

    cfg = get_cfg(market)
    spec = task_table(cfg)[task]
    kind = spec["kind"]
    df = load_predictions(market, period, task, model, fs, start, end)
    if len(df) < 20:
        raise ValueError(f"only {len(df)} evaluated predictions in the selected window (need >= 20)")
    res: dict = {"n": int(len(df)), "start": str(df.index.min().date()), "end": str(df.index.max().date()),
                 "metrics": {k: v for k, v in score(df, kind).items() if not isinstance(v, list)}}
    if kind == "regression":  # price-level errors of the implied price path
        rows = pd.DataFrame(prediction_rows(market, period, task, model, fs, start, end)["rows"])
        a, p = rows["actual_close_target"], rows["predicted_price"]
        ok = a.notna() & p.notna()
        e = (a - p)[ok]
        res["price_metrics"] = {"mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                                "mape_pct": float((e.abs() / a[ok]).mean() * 100)}
    ref = _reference(kind)
    baseline = None
    try:
        ref_df = load_predictions(market, period, task, ref, "price", start, end)
        if model != ref:
            baseline = {"model": ref, "metrics": {k: v for k, v in score(ref_df, kind).items()
                                                  if not isinstance(v, list)},
                        "comparison": paired_test(ref_df, df, kind, spec["horizon"], cfg)}
            res["by_vol_regime_improvement_vs_baseline"] = regime_breakdown(ref_df, df, kind, vol_regimes(market))
    except FileNotFoundError:
        pass
    res["baseline"] = baseline
    # per-regime accuracy of this model
    reg = vol_regimes(market).reindex(df.index)
    by = {}
    for name, g in df.groupby(reg):
        if len(g) >= 20:
            by[name] = {k: v for k, v in score(g, kind).items() if not isinstance(v, list)}
    res["by_vol_regime"] = by
    res["vol_regime_definition"] = "ex-ante 63-session realised vol: low < 12%, mid 12-20%, high > 20% (annualised)"
    return res


# ---- run results ------------------------------------------------------------------------------------
def run_table(market: str, name: str) -> list[dict]:
    run = latest_run(get_cfg(market))
    if run is None or not (run / name).exists():
        raise FileNotFoundError(name)
    p = run / name
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(pd.read_csv(p).to_json(orient="records"))


def run_info(market: str) -> dict | None:
    cfg = get_cfg(market)
    run = latest_run(cfg)
    if run is None:
        return None
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    files = {p.name for p in run.iterdir() if p.is_file()}
    lock = cfg.artifacts_dir / "holdout_lock.json"
    return {"name": run.name, "created_utc": man["created_utc"], "dataset_version": man["dataset_version"],
            "config_fingerprint": man["config_fingerprint"], "feature_sets": man["feature_sets"],
            "stages_complete": {"comparison": "comparison_dev.csv" in files, "ablation": "ablation_dev.csv" in files,
                                "selection": "selection.json" in files, "holdout": "holdout.csv" in files},
            "holdout_evaluations": json.loads(lock.read_text()) if lock.exists() else [],
            "validation": cfg.validation.model_dump()}


def status(market: str) -> dict:
    cfg = get_cfg(market)
    log = DataStore(cfg.data_dir).fetch_log()
    latest = log.sort_values("fetched_at").groupby(["source", "dataset"]).tail(1)
    _, meta = features(market)
    cov = pd.DataFrame(meta["coverage"]).T
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    warnings = []
    for c, r in cov.iterrows():
        last = r["last_valid"]
        if last and (now - pd.Timestamp(last)).days > 10:
            warnings.append({"feature": c, "category": r["category"], "last_valid": last,
                             "days_stale": int((now - pd.Timestamp(last)).days)})
    px = prices(market)
    return {"market": market, "sources": json.loads(latest.to_json(orient="records")),
            "price_coverage": {"start": str(px.index.min().date()), "end": str(px.index.max().date()),
                               "rows": int(len(px))},
            "feature_dataset": {k: meta[k] for k in ("dataset_version", "rows", "first_date", "last_date")},
            "stale_features": warnings, "run": run_info(market),
            "forecast_available": (cfg.artifacts_dir / "predictions" / f"latest_{market}.json").exists()}


def feature_catalog(market: str) -> list[dict]:
    _, meta = features(market)
    return [{"feature": c, **meta["coverage"][c], **meta["provenance"].get(c, {})} for c in meta["coverage"]]


def latest_forecast(market: str) -> dict | None:
    cfg = get_cfg(market)
    p = cfg.artifacts_dir / "predictions" / f"latest_{market}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cors_origins() -> list[str]:
    return [o.strip() for o in os.environ.get("MARKETPRED_CORS_ORIGINS",
                                              "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
