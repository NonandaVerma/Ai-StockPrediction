"""Technical (price/volume) features shared by every model in the dashboard's forecast engine
(src/forecast_engine.py builds the astro and target columns separately and combines them)."""
import numpy as np
import pandas as pd


def technical_features(px: pd.DataFrame) -> pd.DataFrame:
    c = px["close"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=px.index)

    # --- Previously implemented: close-price-only features (returns, volatility, momentum, trend) ---
    f["ret_1"] = r
    f["ret_5"] = r.rolling(5).sum()
    f["ret_20"] = r.rolling(20).sum()
    f["vol_20"] = r.rolling(20).std()
    f["vol_ratio"] = r.rolling(5).std() / f["vol_20"]
    gain = r.clip(lower=0).rolling(14).mean()
    loss = (-r.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    f["dist_ma50"] = c / c.rolling(50).mean() - 1

    # --- New: candlestick geometry + volume features, added to strengthen the technical
    # baseline. All values use only day-t's own OHLCV, so no look-ahead.
    # Only computed when full OHLCV is available: the dashboard's forecast_engine.py calls
    # this with a close-only frame (it handles volume separately with its own logic).
    if {"open", "high", "low", "volume"}.issubset(px.columns):
        o, h, l, v = px["open"], px["high"], px["low"], px["volume"]
        f["gap_pct"] = (o - c.shift(1)) / c.shift(1)
        f["intraday_range_pct"] = (h - l) / c
        f["body_ratio"] = (c - o).abs() / c
        f["upper_wick_ratio"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / c
        f["lower_wick_ratio"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / c
        # A flat bar (high == low) has no "location within the range" - default to the
        # neutral midpoint (0.5) instead of leaving it undefined, same pattern as
        # relative_volume_20 below.
        f["close_location"] = ((c - l) / (h - l).replace(0, np.nan)).fillna(0.5)
        # Some tickers (e.g. index symbols like ^NSEI) report volume=0 for long historical
        # stretches; fill those as "normal" (1.0) instead of NaN so dropna() doesn't discard
        # otherwise-valid price rows just because volume data is missing.
        f["relative_volume_20"] = (v / v.rolling(20).mean()).fillna(1.0)

    # --- New: mean-reversion, drawdown, statistical-behavior and lagged-context features.
    # Close-price-only (works for both the dashboard's close-only and full-OHLCV calls).
    f["price_zscore_20"] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    f["drawdown_60"] = c / c.rolling(60).max() - 1
    f["ret_skew_20"] = r.rolling(20).skew()
    with np.errstate(invalid="ignore", divide="ignore"):
        f["ret_autocorr_5"] = r.rolling(60).apply(_autocorr5, raw=True)
    f["ret_1_lag1"] = r.shift(1)
    f["rsi_14_lag1"] = f["rsi_14"].shift(1)

    return f


def _autocorr5(x: np.ndarray) -> float:
    return np.corrcoef(x[:-5], x[5:])[0, 1]
