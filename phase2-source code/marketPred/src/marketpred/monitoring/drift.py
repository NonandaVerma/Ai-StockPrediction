"""Feature drift (population stability index) and live forecast-vs-outcome tracking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.metrics import binary_metrics


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index with quantile bins from the reference sample.
    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 large shift."""
    ref, cur = reference.dropna().to_numpy(), current.dropna().to_numpy()
    if len(ref) < bins * 5 or len(cur) < 10:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(((c - r) * np.log(c / r)).sum())


def drift_report(X: pd.DataFrame, recent_days: int = 63, reference_years: int = 5) -> pd.DataFrame:
    end = X.index[-1]
    cur = X.iloc[-recent_days:]
    ref = X[(X.index < cur.index[0]) & (X.index >= end - pd.DateOffset(years=reference_years))]
    rows = [{"feature": c, "category": c.split("__")[0], "psi": psi(ref[c], cur[c]),
             "missing_recent": float(cur[c].isna().mean())} for c in X.columns]
    df = pd.DataFrame(rows).sort_values("psi", ascending=False)
    df["status"] = pd.cut(df["psi"], [-np.inf, 0.1, 0.25, np.inf], labels=["stable", "moderate", "large"])
    return df


def rolling_oos_performance(pred: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Rolling AUC/Brier of an out-of-sample binary prediction frame (columns y, pred)."""
    out = []
    for end in range(window, len(pred) + 1, 21):
        w = pred.iloc[end - window:end]
        m = binary_metrics(w["y"].to_numpy(), w["pred"].to_numpy())
        out.append({"date": w.index[-1], "auc": m["auc"], "brier": m["brier"], "base_rate": m["base_rate"]})
    return pd.DataFrame(out).set_index("date")


def reconcile_forecast_log(log_path: Path, targets: pd.DataFrame, tasks: dict) -> pd.DataFrame:
    """Join logged live forecasts with outcomes once they are observed (paper-trading record)."""
    if not log_path.exists():
        return pd.DataFrame()
    log = pd.read_csv(log_path)
    log["session"] = pd.to_datetime(log["session"])
    rows = []
    for _, r in log.iterrows():
        spec = tasks.get(r["task"])
        if spec is None or r["session"] not in targets.index:
            continue
        y = targets.loc[r["session"], spec["target"]]
        rows.append({**r.to_dict(), "outcome": None if pd.isna(y) else float(y),
                     "prediction": json.loads(r["prediction"])})
    return pd.DataFrame(rows)
