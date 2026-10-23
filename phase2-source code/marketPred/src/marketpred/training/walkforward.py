"""Expanding-window walk-forward validation with purging and embargo.

For a label covering sessions i+1..i+h, a training row i is admissible for a test block starting at
position s only if i + h < s - embargo (purging; Lopez de Prado 2018, ch. 7). Test rows of the
development period whose label window would reach into the holdout period are dropped, so the
holdout stays untouched until the final evaluation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..evaluation.metrics import obs_loss
from ..logging_utils import get_logger
from ..models.calibration import PlattCalibrator
from ..models.factory import FEATURE_FREE, HAR_COLS, make_model

log = get_logger("walkforward")


@dataclass
class Fold:
    train: np.ndarray   # integer positions
    test: np.ndarray


def walk_forward_folds(dates: pd.DatetimeIndex, start: str, end: str | None, horizon: int, retrain_every: int,
                       embargo: int, min_train: int, label_limit: str | None = None) -> list[Fold]:
    """Folds over positions of `dates` with test dates in [start, end).

    label_limit: test rows i with label end i+h at or beyond this date are excluded.
    """
    dates = pd.DatetimeIndex(dates)
    n = len(dates)
    s0 = int(dates.searchsorted(pd.Timestamp(start)))
    e0 = int(dates.searchsorted(pd.Timestamp(end))) if end else n
    lim = int(dates.searchsorted(pd.Timestamp(label_limit))) if label_limit else n + horizon
    folds = []
    for s in range(s0, e0, retrain_every):
        test = np.arange(s, min(s + retrain_every, e0))
        test = test[test + horizon < lim]
        train = np.arange(0, max(0, s - horizon - embargo))
        if len(test) and len(train) >= min_train:
            folds.append(Fold(train, test))
    return folds


def calibration_split(train: np.ndarray, frac: float, horizon: int, embargo: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a training window into (fit, calibration) with a purge gap between them."""
    n_cal = int(len(train) * frac)
    cal = train[len(train) - n_cal:]
    fit = train[: max(0, len(train) - n_cal - horizon - embargo)]
    return fit, cal


def model_inputs(model_name: str, feature_cols: list[str], all_cols: list[str]) -> list[str]:
    if model_name in ("rw_vol", "har"):
        return HAR_COLS
    if model_name == "ar1":
        return ["price__ret_1"]
    if model_name in FEATURE_FREE or model_name == "garch":
        return feature_cols[:1] or all_cols[:1]
    return feature_cols


def run_walk_forward(X: pd.DataFrame, y: pd.Series, task: dict, model_name: str, feature_cols: list[str],
                     cfg: Config, period: str = "dev", context: dict | None = None,
                     calibrate: bool | None = None) -> pd.DataFrame:
    """Out-of-sample predictions for one (task, model, feature set). Returns a frame indexed by date."""
    v = cfg.validation
    if period == "dev":
        start, end, limit = v.dev_start, v.holdout_start, v.holdout_start
    elif period == "holdout":
        start, end, limit = v.holdout_start, None, None
    else:
        raise ValueError(period)
    h, kind = task["horizon"], task["kind"]
    folds = walk_forward_folds(X.index, start, end, h, v.retrain_every_days, v.embargo_days, v.min_train_obs, limit)
    cols = model_inputs(model_name, feature_cols, list(X.columns))
    params = cfg.models.params.get(model_name, {})
    if kind == "multiclass" and model_name == "lgbm":
        params = {**params, "objective": "multiclass"}
    if calibrate is None:
        calibrate = kind == "binary" and model_name in cfg.models.calibrate
    grid = cfg.models.tuning.get(model_name, {})
    yv = y.to_numpy()

    def tune(fit_on, grid: dict, base: dict, fit_idx: np.ndarray, val_idx: np.ndarray) -> dict:
        """Pick the grid point with the lowest primary loss on the purged tail of the training window."""
        best, best_loss = base, np.inf
        for combo in itertools.product(*grid.values()):
            cand = {**base, **dict(zip(grid, combo))}
            pr = fit_on(fit_idx, cand).predict(X.iloc[val_idx][cols])
            loss = obs_loss(kind, yv[val_idx], pr).mean()
            if loss < best_loss:
                best, best_loss = cand, loss
        return best

    out = []
    for k, fold in enumerate(folds):
        tr = fold.train[~np.isnan(yv[fold.train])]
        te = fold.test[~np.isnan(yv[fold.test])]
        if len(te) == 0:
            continue

        def fit_on(idx: np.ndarray, p: dict = params):
            m = make_model(model_name, kind, p, h, context, cfg.project.seed, cfg.features.winsorize_quantile)
            return m.fit(X.iloc[idx][cols], y.iloc[idx])

        fit_idx, cal_idx = calibration_split(tr, v.calibration_fraction, h, v.embargo_days)
        fold_params = params
        if grid:
            fold_params = tune(fit_on, grid, params, fit_idx, cal_idx)
        model = fit_on(tr, fold_params)
        pred = model.predict(X.iloc[te][cols])
        if calibrate:
            m_cal = fit_on(fit_idx, fold_params)
            cal = PlattCalibrator().fit(m_cal.predict(X.iloc[cal_idx][cols]), yv[cal_idx])
            pred = cal.transform(pred)
        df = pd.DataFrame(index=X.index[te])
        df["y"] = yv[te]
        if kind == "multiclass":
            for c in range(pred.shape[1]):
                df[f"p{c}"] = pred[:, c]
        else:
            df["pred"] = pred
        df["fold"] = k
        df["params"] = str({k2: fold_params[k2] for k2 in grid}) if grid else ""
        df["train_end"] = X.index[tr[-1]]
        df["train_mean"] = float(np.mean(yv[tr])) if kind != "multiclass" else np.nan
        out.append(df)
    if not out:
        raise RuntimeError(f"no folds for {task['task']} {model_name} {period}")
    res = pd.concat(out)
    res.attrs.update({"task": task["task"], "model": model_name, "period": period, "n_folds": len(out),
                      "calibrated": bool(calibrate)})
    return res


def fit_final_and_predict(X: pd.DataFrame, y: pd.Series, task: dict, model_name: str, feature_cols: list[str],
                          cfg: Config, context: dict | None = None, as_of: pd.Timestamp | None = None) -> dict:
    """Fit on every row whose label is fully observed by `as_of`, then forecast for `as_of`.

    Rows after as_of - horizon have incomplete labels (NaN) and are excluded automatically; the purge
    gap for calibration is applied exactly as in walk-forward validation.
    """
    v, h, kind = cfg.validation, task["horizon"], task["kind"]
    as_of = as_of or X.index[-1]
    pos = int(X.index.get_loc(as_of))
    yv = y.to_numpy()
    tr = np.arange(0, max(0, pos - h - v.embargo_days + 1))
    tr = tr[~np.isnan(yv[tr])]
    cols = model_inputs(model_name, feature_cols, list(X.columns))
    params = dict(cfg.models.params.get(model_name, {}))
    if kind == "multiclass" and model_name == "lgbm":
        params["objective"] = "multiclass"
    calibrate = kind == "binary" and model_name in cfg.models.calibrate

    def fit_on(idx: np.ndarray, p: dict = params):
        m = make_model(model_name, kind, p, h, context, cfg.project.seed, cfg.features.winsorize_quantile)
        return m.fit(X.iloc[idx][cols], y.iloc[idx])

    fit_idx, cal_idx = calibration_split(tr, v.calibration_fraction, h, v.embargo_days)
    grid = cfg.models.tuning.get(model_name, {})
    if grid:
        best, best_loss = params, np.inf
        for combo in itertools.product(*grid.values()):
            cand = {**params, **dict(zip(grid, combo))}
            loss = obs_loss(kind, yv[cal_idx], fit_on(fit_idx, cand).predict(X.iloc[cal_idx][cols])).mean()
            if loss < best_loss:
                best, best_loss = cand, loss
        params = best
    model = fit_on(tr, params)
    x_now = X.iloc[[pos]][cols]
    pred = model.predict(x_now)
    if calibrate:
        cal = PlattCalibrator().fit(fit_on(fit_idx, params).predict(X.iloc[cal_idx][cols]), yv[cal_idx])
        pred = cal.transform(pred)
    return {"prediction": pred[0].tolist() if kind == "multiclass" else float(pred[0]),
            "n_train": int(len(tr)), "train_end": str(X.index[tr[-1]].date()), "calibrated": bool(calibrate),
            "params": {k: params[k] for k in grid} if grid else {}}
