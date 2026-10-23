"""Category A - market prices and trading activity (baseline category).

All features at date t use closes/highs/lows/volume up to and including t (known at the close).
Open prices are not used: before ~2007 Yahoo's ^GSPC opens frequently equal the prior close,
so overnight-gap features would be artefacts of the data vendor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FeatureContext

CAT = "price"
ANN = np.sqrt(252)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def compute(prices: pd.DataFrame) -> pd.DataFrame:
    """Pure function of an OHLCV frame (testable without a context)."""
    c, h, lo = prices["close"], prices["high"], prices["low"]
    r = np.log(c).diff()
    f = {}
    for k in (1, 5, 21, 63, 126, 252):
        f[f"ret_{k}"] = np.log(c / c.shift(k))
    f["mom_12_1"] = f["ret_252"] - f["ret_21"]
    for k in (5, 21, 63):
        f[f"rv_{k}"] = r.rolling(k).std() * ANN
    f["log_rv_5"], f["log_rv_21"], f["log_rv_63"] = (np.log(f[k]) for k in ("rv_5", "rv_21", "rv_63"))
    f["downside_vol_21"] = np.sqrt((r.clip(upper=0) ** 2).rolling(21).mean() * 252)
    f["vol_ratio_5_63"] = f["rv_5"] / f["rv_63"]
    hl = np.log(h / lo)
    f["parkinson_21"] = np.sqrt((hl ** 2).rolling(21).mean() / (4 * np.log(2)) * 252)
    f["range_1"] = hl
    f["dist_high_252"] = np.log(c / c.rolling(252).max())
    f["dist_low_252"] = np.log(c / c.rolling(252).min())
    f["ma_ratio_50"] = np.log(c / c.rolling(50).mean())
    f["ma_ratio_200"] = np.log(c / c.rolling(200).mean())
    f["rsi_14"] = rsi(c)
    f["skew_63"] = r.rolling(63).skew()
    thresh = 3 * r.rolling(63).std().shift()
    f["jumps_63"] = (r.abs() > thresh).astype(float).rolling(63).sum()
    v = np.log(prices["volume"].replace(0, np.nan))
    f["volume_z_63"] = (v - v.rolling(63, min_periods=40).mean()) / v.rolling(63, min_periods=40).std()
    f["volume_trend_5_63"] = v.rolling(5).mean() - v.rolling(63, min_periods=40).mean()
    return pd.DataFrame(f, index=prices.index)


def build(ctx: FeatureContext) -> pd.DataFrame:
    feats = compute(ctx.prices).reindex(ctx.dates)
    feats.columns = [ctx.register(CAT, c, feats[c], "yahoo:" + ctx.cfg.market.ticker, "see price.compute")
                     for c in feats.columns]
    return feats
