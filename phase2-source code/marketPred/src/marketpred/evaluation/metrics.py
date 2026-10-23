"""Task-appropriate forecast metrics and per-observation losses (for paired statistical tests)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)

EPS = 1e-6


def logloss_obs(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier_obs(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (p - y) ** 2


def multiclass_logloss_obs(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    P = np.clip(P, EPS, 1)
    P = P / P.sum(axis=1, keepdims=True)
    return -np.log(P[np.arange(len(y)), y.astype(int)])


def multiclass_brier_obs(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y.astype(int)] = 1
    return ((P - Y) ** 2).sum(axis=1)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Equal-frequency-bin ECE: weighted mean |observed frequency - mean forecast|."""
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return float(sum(len(c) / len(p) * abs(y[c].mean() - p[c].mean()) for c in chunks if len(c)))


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    order = np.argsort(p)
    return [{"mean_forecast": float(p[c].mean()), "observed": float(y[c].mean()), "n": int(len(c))}
            for c in np.array_split(order, bins) if len(c)]


def binary_metrics(y: np.ndarray, p: np.ndarray, base_rate: float | None = None) -> dict:
    y = y.astype(int)
    pred = (p > 0.5).astype(int)
    m = {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "log_loss": float(logloss_obs(y, p).mean()),
        "brier": float(brier_obs(y, p).mean()),
        "ece": expected_calibration_error(y, p),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float((pred == y).mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mean_forecast": float(p.mean()),
    }
    if base_rate is not None:  # Brier skill score vs constant forecast
        ref = brier_obs(y, np.full(len(y), base_rate)).mean()
        m["brier_skill"] = float(1 - m["brier"] / ref) if ref > 0 else float("nan")
    return m


def regression_metrics(y: np.ndarray, f: np.ndarray, benchmark: np.ndarray | None = None) -> dict:
    e = y - f
    m = {
        "n": int(len(y)),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "corr": float(np.corrcoef(y, f)[0, 1]) if np.std(f) > 0 else float("nan"),
        "hit_rate": float((np.sign(y) == np.sign(f)).mean()),
    }
    if benchmark is not None:  # Campbell-Thompson out-of-sample R^2 vs a benchmark forecast
        m["r2_oos"] = float(1 - (e ** 2).sum() / ((y - benchmark) ** 2).sum())
    return m


def qlike_obs(log_rv_true: np.ndarray, log_rv_pred: np.ndarray) -> np.ndarray:
    """QLIKE loss on variances (Patton 2011), robust to noise in the volatility proxy."""
    ratio = np.exp(2 * (log_rv_true - log_rv_pred))
    return ratio - np.log(ratio) - 1


def volatility_metrics(y: np.ndarray, f: np.ndarray) -> dict:
    e = y - f
    mz = np.polyfit(f, y, 1) if np.std(f) > 0 else [np.nan, np.nan]
    r2 = np.corrcoef(y, f)[0, 1] ** 2 if np.std(f) > 0 else float("nan")
    return {
        "n": int(len(y)),
        "mse_log": float((e ** 2).mean()),
        "mae_log": float(np.abs(e).mean()),
        "qlike": float(qlike_obs(y, f).mean()),
        "mz_slope": float(mz[0]), "mz_intercept": float(mz[1]), "mz_r2": float(r2),
        "bias_log": float(e.mean()),
    }


def multiclass_metrics(y: np.ndarray, P: np.ndarray) -> dict:
    pred = P.argmax(axis=1)
    return {
        "n": int(len(y)),
        "log_loss": float(multiclass_logloss_obs(y, P).mean()),
        "brier": float(multiclass_brier_obs(y, P).mean()),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y.astype(int), pred)),
        "class_freq": np.bincount(y.astype(int), minlength=P.shape[1]).tolist(),
    }


def obs_loss(kind: str, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Primary per-observation loss used for model comparison tests."""
    if kind == "binary":
        return logloss_obs(y, pred)
    if kind == "multiclass":
        return multiclass_logloss_obs(y, pred)
    return (y - pred) ** 2  # regression and log-volatility: squared error


def summarize(kind: str, y: np.ndarray, pred: np.ndarray, base: float | np.ndarray | None = None) -> dict:
    if kind == "binary":
        return binary_metrics(y, pred, base_rate=base if np.isscalar(base) else None)
    if kind == "multiclass":
        return multiclass_metrics(y, pred)
    if kind == "volatility":
        return volatility_metrics(y, pred)
    return regression_metrics(y, pred, benchmark=base if base is not None and not np.isscalar(base) else None)
