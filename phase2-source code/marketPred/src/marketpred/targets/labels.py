"""Prediction targets. All are defined from the close of decision day t (index position i).

Let C_i be the close on session i and r_i = ln(C_i / C_{i-1}). For horizon h (trading sessions):

  fwd_ret_h[i]  = ln(C_{i+h} / C_i)                                    (Target 2, expected return)
  dir_h[i]      = 1{fwd_ret_h[i] > 0}                                  (Target 1, direction)
  fwd_rv_h[i]   = sqrt(252/h * sum_{k=1..h} r_{i+k}^2)                 (Target 3, annualised realised vol)
  log_rv_h[i]   = ln(fwd_rv_h[i])                                      (modelled vol target)
  down_h[i]     = 1{fwd_ret_h[i] < threshold_h}                        (Target 4, downside risk)
  large_move[i] = 1{|fwd_ret_1[i]| > abs_threshold}                    (Target 6, event/large-move risk)
  regime[i]     = high_vol (3) if fwd_rv_H > high_vol_annualized,      (Target 5, regime over H days)
                  else bull (0) if fwd_ret_H > bull_return,
                  else bear (1) if fwd_ret_H < bear_return,
                  else sideways (2)

The label of row i depends on sessions i+1..i+h only; label_end_h[i] = date of session i+h is used to
purge overlapping training labels. Rows whose window extends past the last available close are NaN.
Prices are ^GSPC price-index closes (dividends excluded).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TargetsCfg

REGIMES = ["bull", "bear", "sideways", "high_vol"]


def forward_sum(x: pd.Series, h: int) -> pd.Series:
    """sum_{k=1..h} x[i+k], NaN where the window is incomplete."""
    return x[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1)


def build_targets(close: pd.Series, cfg: TargetsCfg) -> pd.DataFrame:
    close = close.astype(float)
    r = np.log(close).diff()
    out = pd.DataFrame(index=close.index)
    horizons = sorted(set(cfg.horizons) | set(cfg.vol_horizons) | set(cfg.downside) | {int(cfg.regime["horizon"])})
    for h in horizons:
        fr = np.log(close.shift(-h) / close)
        out[f"fwd_ret_{h}"] = fr
        out[f"dir_{h}"] = (fr > 0).astype(float).where(fr.notna())
        rv = np.sqrt(252.0 / h * forward_sum(r ** 2, h))
        out[f"fwd_rv_{h}"] = rv
        out[f"log_rv_{h}"] = np.log(rv.where(rv > 0))
        out[f"label_end_{h}"] = pd.Series(close.index, index=close.index).shift(-h)
    for h, thr in cfg.downside.items():
        fr = out[f"fwd_ret_{h}"]
        out[f"down_{h}"] = (fr < thr).astype(float).where(fr.notna())
    lm_h = int(cfg.large_move["horizon"])
    fr = out[f"fwd_ret_{lm_h}"]
    out["large_move"] = (fr.abs() > cfg.large_move["abs_threshold"]).astype(float).where(fr.notna())
    H = int(cfg.regime["horizon"])
    fr, rv = out[f"fwd_ret_{H}"], out[f"fwd_rv_{H}"]
    reg = np.where(rv > cfg.regime["high_vol_annualized"], 3,
                   np.where(fr > cfg.regime["bull_return"], 0, np.where(fr < cfg.regime["bear_return"], 1, 2)))
    out["regime"] = pd.Series(reg, index=close.index, dtype=float).where(fr.notna() & rv.notna())
    return out


def task_specs(cfg: TargetsCfg) -> list[dict]:
    """Enumerate forecasting tasks: name, kind, target column, horizon (for purging)."""
    tasks = [{"task": f"direction_{h}", "kind": "binary", "target": f"dir_{h}", "horizon": h} for h in cfg.horizons]
    tasks += [{"task": f"return_{h}", "kind": "regression", "target": f"fwd_ret_{h}", "horizon": h}
              for h in cfg.horizons]
    tasks += [{"task": f"volatility_{h}", "kind": "volatility", "target": f"log_rv_{h}", "horizon": h}
              for h in cfg.vol_horizons]
    tasks += [{"task": f"downside_{h}", "kind": "binary", "target": f"down_{h}", "horizon": h} for h in cfg.downside]
    tasks.append({"task": "large_move_1", "kind": "binary", "target": "large_move",
                  "horizon": int(cfg.large_move["horizon"])})
    tasks.append({"task": f"regime_{int(cfg.regime['horizon'])}", "kind": "multiclass", "target": "regime",
                  "horizon": int(cfg.regime["horizon"])})
    return tasks
