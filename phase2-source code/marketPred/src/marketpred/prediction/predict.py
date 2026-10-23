"""Forecast generation: structured, versioned forecast records for the latest decision date.

Each record states what it is (a forecast, not an outcome), the model/data versions, feature coverage,
data-quality warnings, and the validation evidence behind each number. Probabilities are labelled
"calibrated" only when Platt calibration was applied and validated out of sample.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import Config
from ..features.build import columns_for
from ..logging_utils import get_logger, log_event
from ..targets.labels import REGIMES
from ..training.experiments import Dataset, load_dataset
from ..training.walkforward import fit_final_and_predict

log = get_logger("predict")


def latest_run(cfg: Config) -> Path:
    runs = sorted((cfg.artifacts_dir / "runs").glob("*/selection.json"))
    if not runs:
        raise FileNotFoundError("no run with selection.json; run `marketpred evaluate` first")
    return runs[-1].parent


def data_quality(ds: Dataset, as_of: pd.Timestamp, cols: list[str]) -> tuple[dict, list[str]]:
    row = ds.X.loc[as_of, cols]
    coverage = {}
    for cat in sorted({c.split("__")[0] for c in cols}):
        cc = [c for c in cols if c.startswith(cat + "__")]
        coverage[cat] = float(row[cc].notna().mean())
    warnings = [f"{c} missing at forecast date (stale or unavailable source)" for c in cols if pd.isna(row[c])]
    age = (pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - as_of).days
    if age > 4:
        warnings.append(f"latest price data is {age} days old; run `marketpred ingest`")
    return coverage, warnings


def holdout_metrics(run_dir: Path) -> dict:
    path = run_dir / "holdout.csv"
    if not path.exists():
        return {}
    h = pd.read_csv(path)
    return {t: g[g["role"] == "selected"].drop(columns=["task"]).iloc[0].dropna().to_dict()
            for t, g in h.groupby("task") if (g["role"] == "selected").any()}


def make_forecast(cfg: Config, run_dir: Path | None = None, as_of: str | None = None) -> dict:
    run_dir = run_dir or latest_run(cfg)
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))["tasks"]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    ds = load_dataset(cfg)
    t_now = pd.Timestamp(as_of) if as_of else ds.X.index[-1]
    from ..calendar import decision_times

    decision_ts = decision_times(pd.DatetimeIndex([t_now]), cfg.market.calendar).iloc[0]
    validation = holdout_metrics(run_dir)
    tasks_out, used_cols = {}, set()
    for task, s in selection.items():
        spec = ds.tasks[task]
        cats = (s["combined_feature_set"] if s["best_dev_feature_set"].startswith("combined")
                else manifest["feature_sets"].get(s["best_dev_feature_set"], cfg.features.categories))
        cols = columns_for(ds.X.columns, cats)
        used_cols.update(cols)
        res = fit_final_and_predict(ds.X, ds.T[spec["target"]], spec, s["best_dev_model"], cols, cfg, ds.context,
                                    t_now)
        tasks_out[task] = {"model": s["best_dev_model"], "feature_set": s["best_dev_feature_set"],
                           "categories": cats, **res, "validation_holdout": validation.get(task, {})}
        log_event(log, "forecast", task=task, model=s["best_dev_model"], pred=res["prediction"])
    coverage, warnings = data_quality(ds, t_now, sorted(used_cols))

    def get(task: str):
        return tasks_out.get(task, {}).get("prediction")

    horizons = {}
    for h in cfg.targets.horizons:
        p_up = get(f"direction_{h}")
        exp_ret = get(f"return_{h}")
        # Use the same-horizon vol forecast, else the nearest horizon's annualised forecast (labelled).
        vol_tasks = sorted((abs(int(t.split("_")[1]) - h), t) for t in tasks_out if t.startswith("volatility_"))
        vol_task = vol_tasks[0][1] if vol_tasks else None
        vol_ann = float(np.exp(get(vol_task))) if vol_task else None
        sigma_h = vol_ann * np.sqrt(h / 252) if vol_ann else None
        horizons[str(h)] = {
            "prob_positive_return": p_up,
            # "platt" only if a calibration map was fitted; otherwise judge by the validation ECE/Brier.
            "prob_calibration": "platt" if tasks_out.get(f"direction_{h}", {}).get("calibrated")
            else "none (see validation_holdout ece)",
            "expected_log_return": exp_ret,
            "return_uncertainty_sd": sigma_h,
            "return_interval_90": [exp_ret + norm.ppf(0.05) * sigma_h, exp_ret + norm.ppf(0.95) * sigma_h]
            if (exp_ret is not None and sigma_h) else None,
            "predicted_volatility_annualized": vol_ann,
            "volatility_source_task": vol_task,
            "value_at_risk_95": float(-norm.ppf(0.05) * sigma_h) if sigma_h else None,
            "prob_loss_beyond_threshold": get(f"downside_{h}"),
            "loss_threshold": cfg.targets.downside.get(h),
        }
    regime = get(f"regime_{int(cfg.targets.regime['horizon'])}")
    record = {
        "record_type": "forecast",  # not an observed outcome, not a backtest
        "asset": cfg.market.ticker, "market": cfg.market.id, "market_name": cfg.market.name,
        "prediction_timestamp_utc": decision_ts.isoformat(),
        "data_as_of_session": str(t_now.date()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "horizons_trading_days": horizons,
        "large_move_probability_1d": get("large_move_1"),
        "regime_horizon_days": int(cfg.targets.regime["horizon"]),
        "regime_probabilities": dict(zip(REGIMES, regime)) if regime else None,
        "regime_definitions": cfg.targets.regime,
        "versions": {"dataset_version": manifest["dataset_version"], "config_fingerprint": cfg.fingerprint(),
                     "run": run_dir.name, "package": __import__("marketpred").__version__},
        "feature_coverage": coverage,
        "data_quality_warnings": warnings,
        "tasks": tasks_out,
        "disclaimer": "Research output. Out-of-sample evidence is reported per task in validation_holdout; "
                      "direction forecasts showed no reliable skill. Not investment advice.",
    }
    out_dir = cfg.artifacts_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"forecast_{cfg.market.id}_{t_now.date()}.json").write_text(
        json.dumps(record, indent=2, default=float), encoding="utf-8")
    (out_dir / f"latest_{cfg.market.id}.json").write_text(json.dumps(record, indent=2, default=float),
                                                          encoding="utf-8")
    _append_log(out_dir / "forecast_log.csv", record)
    return record


def _append_log(path: Path, rec: dict) -> None:
    """Flat log of forecasts so they can be compared with realised outcomes later (paper trading)."""
    rows = []
    for task, t in rec["tasks"].items():
        pred = t["prediction"]
        rows.append({"session": rec["data_as_of_session"], "generated_at_utc": rec["generated_at_utc"],
                     "task": task, "model": t["model"], "feature_set": t["feature_set"],
                     "prediction": json.dumps(pred), "dataset_version": rec["versions"]["dataset_version"]})
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
