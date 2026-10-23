"""Actual vs predicted for a past period ("hindcast").

Two ways to predict the sessions of a past window W = [d_1 .. d_N]:
  strict    one forecast made after the close of the session BEFORE d_1. Prices of every session in W
            are cut off before the model sees anything; day k is the forecast for k sessions ahead.
  rolling   each d_i is predicted after the close of d_{i-1}: one day ahead, using actual prices up
            to the previous day only (never the day being predicted).
In both modes the engine only ever receives `px.loc[:as_of]` (point-in-time), and every prediction is
made by src/training.py with the same settings as the Future Forecast page: strict day k equals a Future
Forecast of the next k sessions made on the same as-of date. Session dates of W are used because exchange calendars are published in advance.

Predicted candles (so the prediction can be drawn exactly like the actual chart):
  close  the model's expected close
  open   the previous predicted close (day 1: the last actual close before the period; rolling mode:
         the previous day's actual close, which is known at the forecast time)
  high/low  the instrument's typical wicks over the CANDLE_LOOKBACK sessions before the forecast date,
         added above max(open, close) and below min(open, close)
The separate 80% forecast range is kept in range_low / range_high.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from .analysis import forward_returns
from .forecast_engine import build_dataset, sky_for_sessions
from .forecast_models import SELECTABLE_GROUPS, SPEC_BY_NAME
from .training import TrainSettings, predict

MAX_SESSIONS = 60
CANDLE_LOOKBACK = 60


def candle_shape(px_pit: pd.DataFrame, n: int = CANDLE_LOOKBACK) -> tuple[float, float]:
    """Mean log upper and lower wick over the last `n` sessions (data up to the forecast date only)."""
    t = px_pit.tail(n)
    o, h, l, c = (t[k].astype(float) for k in ("open", "high", "low", "close"))
    up = np.log(h / np.maximum(o, c)).clip(lower=0)
    dn = np.log(np.minimum(o, c) / l).clip(lower=0)
    return float(up.mean()), float(dn.mean())


def _needs_extra_groups(model: str) -> bool:
    return any(g not in ("market", "astro") for g in SPEC_BY_NAME[model].groups)


def _dataset(px_pit, cal, s, anchor, n_ahead, model, external_fn, category_ctx):
    future = cal.next_sessions(px_pit.index[-1], n_ahead + 1)
    sky = sky_for_sessions(px_pit.index.append(future), cal, s, anchor)
    if not _needs_extra_groups(model):
        return build_dataset(px_pit, sky, 0, 1)
    external, notes = external_fn(px_pit.index)
    ds = build_dataset(px_pit, sky, 0, 1, external, notes, category_ctx)
    missing = [g for g in SPEC_BY_NAME[model].groups if g != "all_categories" and g not in ds.X]
    if missing or not ds.has_groups(SPEC_BY_NAME[model].groups):
        names = ", ".join(SELECTABLE_GROUPS.get(g, g) for g in missing) or "the selected categories"
        raise ValueError(f"no data for {names} on this instrument" + (f" ({' '.join(ds.notes)})" if ds.notes else ""))
    return ds


def _predict(ds, model, horizon, train: TrainSettings):
    """Expected log return over the next `horizon` sessions from the last row of ds (shared recipe)."""
    f = predict(replace(ds, fwd=forward_returns(ds.px["log_ret"], 0, horizon), horizon=horizon), model, train)
    return {k: f[k] for k in ("prob_up", "expected", "lo", "hi", "n_train")}


def hindcast(px_full: pd.DataFrame, cal, s, window: pd.DatetimeIndex, model: str, mode: str = "strict",
             anchor: str = "open", train: TrainSettings | None = None, external_fn=None, category_ctx=None) -> dict:
    """`window`: past sessions present in px_full. Returns per-session actual and predicted closes."""
    train = train or TrainSettings()
    window = pd.DatetimeIndex(window)
    if len(window) == 0:
        raise ValueError("no trading sessions in the selected period")
    if len(window) > MAX_SESSIONS:
        raise ValueError(f"choose at most {MAX_SESSIONS} sessions")
    pos0 = px_full.index.get_loc(window[0])
    if pos0 == 0:
        raise ValueError("no price history before the period")
    actual = px_full.loc[window]
    rows = []
    if mode == "strict":
        as_of = px_full.index[pos0 - 1]
        px_pit = px_full.loc[:as_of]                            # <- nothing from the period
        ds = _dataset(px_pit, cal, s, anchor, len(window), model, external_fn, category_ctx)
        base_close = float(px_pit["close"].iloc[-1])
        wick_up, wick_dn = candle_shape(px_pit)
        prev = base_close
        for k, d in enumerate(window, start=1):
            r = _predict(ds, model, k, train)
            close = base_close * np.exp(r["expected"])
            rows.append({"date": d, "as_of": as_of, "sessions_ahead": k, "reference_close": base_close,
                         "predicted_open": prev, "wick_up": wick_up, "wick_dn": wick_dn, **r})
            prev = close
    elif mode == "rolling":
        for i, d in enumerate(window):
            as_of = px_full.index[pos0 + i - 1]
            px_pit = px_full.loc[:as_of]                        # <- up to the previous day only
            ds = _dataset(px_pit, cal, s, anchor, 1, model, external_fn, category_ctx)
            r = _predict(ds, model, 1, train)
            wick_up, wick_dn = candle_shape(px_pit)
            ref = float(px_pit["close"].iloc[-1])
            rows.append({"date": d, "as_of": as_of, "sessions_ahead": 1, "reference_close": ref,
                         "predicted_open": ref, "wick_up": wick_up, "wick_dn": wick_dn, **r})
    else:
        raise ValueError(mode)
    out = pd.DataFrame(rows).set_index("date")
    ref = out["reference_close"]
    out["predicted_close"] = ref * np.exp(out["expected"])
    body_hi = np.maximum(out["predicted_open"], out["predicted_close"])
    body_lo = np.minimum(out["predicted_open"], out["predicted_close"])
    out["predicted_high"] = body_hi * np.exp(out["wick_up"])
    out["predicted_low"] = body_lo * np.exp(-out["wick_dn"])
    out["range_low"] = ref * np.exp(out["lo"])                   # 80% forecast range for the close
    out["range_high"] = ref * np.exp(out["hi"])
    out["actual_close"] = actual["close"].astype(float)
    out["error_pct"] = 100 * (out["predicted_close"] / out["actual_close"] - 1)
    out["actual_up"] = out["actual_close"] > ref
    out["predicted_up"] = out["prob_up"] > 0.5
    out["direction_hit"] = out["actual_up"] == out["predicted_up"]
    out["confident"] = (out["prob_up"] - 0.5).abs() > train.neutral_band      # a bullish/bearish call
    sure = out[out["confident"]]
    out["inside_80pct_range"] = out["actual_close"].between(out["range_low"], out["range_high"])
    summary = {"mode": mode, "model": model, "sessions": len(out),
               "mae_pct": float(out["error_pct"].abs().mean()),
               "direction_hit_rate": float(out["direction_hit"].mean()),
               "range_coverage": float(out["inside_80pct_range"].mean()),
               "confident_days": int(len(sure)),
               "confident_hit_rate": float(sure["direction_hit"].mean()) if len(sure) else float("nan"),
               "training": train.describe(),
               "first_as_of": str(out["as_of"].iloc[0].date())}
    return {"actual": actual, "predicted": out, "summary": summary}
