"""The one training recipe used by every page.

Future Forecast, Home, Model Comparison, Backtesting and Actual vs Predicted all score, fit and predict
through these functions with the same TrainSettings, so a model given the same data and as-of date
returns the same numbers on every page. A model's scorecard (walk-forward out-of-sample predictions,
80% range, evidence) depends only on that model and its data, never on which other models were picked.

Settings (config.yaml, `forecast:` section):
  folds               walk-forward folds used to score a model and to size its 80% range
  min_train_sessions  fewer usable training rows => no forecast
  train_window_years  null = every session before the forecast date (expanding window);
                      N = only the last N years (markets change; very old data can hurt)
  min_move            the up/down classifier leaves out rows whose move was smaller than this fraction
                      of the expected move (EWMA daily volatility x sqrt(horizon)): the sign of a
                      near-zero move is mostly noise. 0 = use every row. The return model uses every row.
  neutral_band        |P(up) - 0.5| at or below this => "neutral" (no call); also defines confident calls
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .forecast_models import SPEC_BY_NAME, FittedModel
from .repository import stable_hash
from .validation import walk_forward_splits

PI_LEVEL = 0.80
MIN_FOLD_TRAIN = 100          # walk-forward folds with fewer training rows are skipped


@dataclass(frozen=True)
class TrainSettings:
    folds: int = 6
    min_train_sessions: int = 500
    train_window_years: float | None = None
    min_move: float = 0.0
    neutral_band: float = 0.03
    seed: int = 42
    tune: bool = False     # opt-in: nested walk-forward hyperparameter search for lgbm/xgb (slower)

    @classmethod
    def from_cfg(cls, cfg) -> "TrainSettings":
        fc = cfg.get("forecast") or {}
        w = fc.get("train_window_years")
        return cls(folds=int(fc.get("folds", 6)), min_train_sessions=int(fc.get("min_train_sessions", 500)),
                   train_window_years=float(w) if w else None, min_move=float(fc.get("min_move", 0.0)),
                   neutral_band=float(fc.get("neutral_band", 0.03)), seed=int(cfg.get("seed", 42)),
                   tune=bool(fc.get("tune_hyperparams", False)))

    @property
    def key(self) -> str:
        """Identifies the recipe: part of every cache key and stored model version."""
        return stable_hash(asdict(self))

    def describe(self) -> str:
        win = f"last {self.train_window_years:g} years" if self.train_window_years else "all history"
        move = f", up/down learned from moves >= {self.min_move:g} x expected move" if self.min_move else ""
        tuned = " · lgbm/xgb hyperparameters tuned per fold" if self.tune else ""
        return (f"Training: {win} before the forecast date{move} · {self.folds} walk-forward folds · "
                f"neutral when P(up) is within {self.neutral_band:.0%} of 50%{tuned}")


def _window_start(end: pd.Timestamp, s: TrainSettings):
    return None if not s.train_window_years else end - pd.Timedelta(days=round(365.25 * s.train_window_years))


def _clf_rows(ds, y: np.ndarray, idx: pd.Index, s: TrainSettings):
    if not s.min_move:
        return None
    vol = ds.X["market"]["ewma_vol"].reindex(idx).to_numpy(float)
    return np.abs(y) >= s.min_move * vol * np.sqrt(ds.horizon)


def _xy(ds, model: str, X_override=None):
    X = X_override if X_override is not None else ds.matrix(SPEC_BY_NAME[model].groups)
    rows = ds.train_mask & X.notna().all(axis=1)          # external groups can start later than prices
    Xr = X.loc[rows]
    return X, Xr.to_numpy(float), ds.fwd[rows].to_numpy(), Xr.index


def walk_forward(ds, model: str, s: TrainSettings, X_override=None) -> pd.DataFrame:
    """Out-of-sample predictions: expanding (or trailing-window) training, never shuffled, with an
    embargo of gap+horizon rows between training and test so target windows cannot overlap."""
    spec = SPEC_BY_NAME[model]
    _, X, y, idx = _xy(ds, model, X_override)
    clf_ok = _clf_rows(ds, y, idx, s)
    embargo = ds.gap + ds.horizon
    p_up, exp = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for tr, te in walk_forward_splits(len(y), s.folds, embargo):
        tr, te = np.asarray(tr), np.asarray(te)
        start = _window_start(idx[te[0]], s) if len(te) else None
        if start is not None:
            tr = tr[idx[tr] >= start]
        if len(tr) < MIN_FOLD_TRAIN:
            continue
        fm = FittedModel(spec, s.seed, s.tune).fit(X[tr], y[tr], None if clf_ok is None else clf_ok[tr], embargo)
        p_up[te], exp[te] = fm.predict(X[te])
    out = pd.DataFrame({"p_up": p_up, "expected": exp, "actual": y}, index=idx).dropna()
    out.index.name = "date"
    return out


def fit_final(ds, model: str, s: TrainSettings):
    """Fit on every usable row up to the as-of session (or the trailing window). Returns (model, X, n)."""
    spec = SPEC_BY_NAME[model]
    X, Xa, y, idx = _xy(ds, model)
    start = _window_start(ds.px.index[-1], s)
    keep = np.ones(len(y), bool) if start is None else np.asarray(idx >= start)
    clf_ok = _clf_rows(ds, y, idx, s)
    fm = FittedModel(spec, s.seed, s.tune).fit(Xa[keep], y[keep], None if clf_ok is None else clf_ok[keep],
                                               ds.gap + ds.horizon)
    return fm, X, int(keep.sum())


def usable_rows(ds, s: TrainSettings) -> int:
    """Training rows available to the market-only models (the minimum every forecast needs)."""
    m = ds.train_mask
    start = _window_start(ds.px.index[-1], s)
    return int((m & (ds.px.index >= start)).sum()) if start is not None else int(m.sum())


def predict(ds, model: str, s: TrainSettings, oof: pd.DataFrame | None = None) -> dict:
    """Forecast from the last row of `ds` (the as-of session): P(up), expected log return over the
    target window and an 80% range from the model's own out-of-sample errors."""
    fm, X, n = fit_final(ds, model, s)
    if n < s.min_train_sessions:
        raise ValueError(f"only {n} usable training sessions before the forecast date "
                         f"(need {s.min_train_sessions})")
    x_now = X.iloc[[-1]]
    if x_now.isna().any(axis=1).iloc[0]:
        raise ValueError("this model's input data is missing on the forecast date")
    p, e = (float(v[0]) for v in fm.predict(x_now.to_numpy(float)))
    oof = walk_forward(ds, model, s) if oof is None else oof
    q = (1 - PI_LEVEL) / 2
    resid = oof["actual"] - oof["expected"]
    return {"prob_up": p, "expected": e, "lo": e + float(resid.quantile(q)), "hi": e + float(resid.quantile(1 - q)),
            "n_train": n, "oof": oof, "fitted": fm, "columns": list(X.columns)}


def direction_of(p: float, band: float) -> str:
    return "bullish" if p > 0.5 + band else "bearish" if p < 0.5 - band else "neutral"
