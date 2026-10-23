"""Market + cross-asset inputs for the risk pipeline, aligned point-in-time to index sessions.

Decision time: after the index close of session t (NSE closes 15:30 IST = 10:00 UTC).
Each external series carries an availability rule relative to that moment:

  same_session   the value dated t is final before the decision (India VIX is computed
                 by NSE during the same session)
  next_session   the value dated t is NOT known yet: the S&P 500, Brent futures and the
                 Yahoo USD/INR bar all close hours after 10:00 UTC. Row t may only use
                 values dated strictly before t.

Missing data is never filled forward beyond `max_stale` sessions; older values become NaN.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")

CROSS_ASSETS = {
    # name: (Yahoo symbol, availability rule)
    "india_vix": ("^INDIAVIX", "same_session"),
    "usd_inr": ("INR=X", "next_session"),
    "brent": ("BZ=F", "next_session"),
    "us_equity": ("^GSPC", "next_session"),
}


def _cache_path(symbol: str, start: str, end: str) -> Path:
    return RAW_DIR / f"{symbol.replace('^', '').replace('=', '_')}_{start}_{end}.csv"


def load_series(symbol: str, start: str, end: str, reuse_newer: bool = False) -> pd.DataFrame:
    """Daily OHLCV before `end` (yfinance's end is exclusive) from the local cache, downloading once if
    missing. `reuse_newer`: a cache that runs past `end` is cut back to `end` instead of downloading
    again (the dashboard asks for a new end every session; pinned research runs keep exact vintages)."""
    cache = _cache_path(symbol, start, end)
    if cache.exists():
        return pd.read_csv(cache, index_col="date", parse_dates=True)
    if reuse_newer:
        newer = sorted(p for p in RAW_DIR.glob(_cache_path(symbol, start, "*").name) if p.stem.rsplit("_", 1)[-1] > end)
        if newer:
            df = pd.read_csv(newer[0], index_col="date", parse_dates=True)
            return df[df.index < pd.Timestamp(end)]
    import yfinance as yf
    raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"No data returned for {symbol}.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df


def align_to_sessions(values: pd.Series, sessions: pd.DatetimeIndex, rule: str,
                      max_stale: int = 5) -> pd.Series:
    """Latest value usable at the close of each session under `rule`."""
    if rule not in ("same_session", "next_session"):
        raise ValueError(rule)
    v = values.dropna().sort_index()
    sessions = pd.DatetimeIndex(sessions).as_unit("ns")
    left = pd.DataFrame({"session": sessions})
    right = pd.DataFrame({"obs_date": pd.DatetimeIndex(v.index).as_unit("ns"), "value": v.to_numpy()})
    m = pd.merge_asof(left, right, left_on="session", right_on="obs_date", direction="backward",
                      allow_exact_matches=(rule == "same_session"))
    # staleness in index sessions between the observation and the decision
    pos_obs = sessions.searchsorted(m["obs_date"].fillna(sessions[0]), side="left")
    stale = np.arange(len(sessions)) - pos_obs
    out = m["value"].where(m["obs_date"].notna() & (stale <= max_stale))
    return pd.Series(out.to_numpy(), index=sessions, name=values.name)


def synthetic_cross_assets(sessions: pd.DatetimeIndex, seed: int) -> dict[str, pd.DataFrame]:
    """Independent random walks (no link to the index) for offline runs."""
    rng = np.random.default_rng(seed + 1)
    out = {}
    for i, name in enumerate(CROSS_ASSETS):
        level = 20.0 if name == "india_vix" else 100.0
        if name == "india_vix":   # mean-reverting positive series
            x = np.empty(len(sessions)); x[0] = np.log(level)
            for t in range(1, len(x)):
                x[t] = x[t - 1] + 0.05 * (np.log(level) - x[t - 1]) + 0.06 * rng.standard_normal()
            close = np.exp(x)
        else:
            close = level * np.exp(np.cumsum(0.01 * rng.standard_normal(len(sessions))))
        out[name] = pd.DataFrame({"close": close}, index=sessions)
    return out


def load_cross_assets(sessions: pd.DatetimeIndex, start: str, end: str, synthetic: bool, seed: int):
    """{name: close series aligned to `sessions`} and the raw frames used (for the manifest)."""
    raw = synthetic_cross_assets(sessions, seed) if synthetic else \
        {n: load_series(sym, start, end) for n, (sym, _) in CROSS_ASSETS.items()}
    aligned = {}
    for name, (_, rule) in CROSS_ASSETS.items():
        # log returns are computed on the series' OWN calendar, then aligned
        close = raw[name]["close"].rename(name)
        aligned[name] = pd.DataFrame({
            "level": align_to_sessions(close, sessions, rule),
            **{f"ret_{k}": align_to_sessions(np.log(close).diff(k), sessions, rule) for k in (1, 5)},
            "vol_20": align_to_sessions(np.log(close).diff().rolling(20).std() * np.sqrt(252), sessions, rule),
        }, index=sessions)
    return aligned, raw


def _mercury_retrograde(dates: pd.DatetimeIndex) -> np.ndarray:
    """Computed directly with pyswisseph (not through src.astro_features, the code under test)."""
    import swisseph as swe
    out = np.empty(len(dates))
    for i, d in enumerate(dates):
        pos, _ = swe.calc_ut(swe.julday(d.year, d.month, d.day, 3.75), swe.MERCURY, swe.FLG_MOSEPH | swe.FLG_SPEED)
        out[i] = pos[3] < 0
    return out


def synthetic_ohlc(start: str, end: str, seed: int, vol_signal: float = 0.0, signal: str = "mercury_retro") -> pd.DataFrame:
    """GARCH-like index with intraday ranges. If vol_signal > 0, a fake astrological *risk* effect
    scales daily volatility:
      mercury_retro  x exp(vol_signal) while Mercury is retrograde (a ~3-week condition, the kind a
                     20-session risk target can detect)
      lunar          x exp(vol_signal * cos(moon phase)) (mean synodic month). Over a 20-session window
                     this averages out to a ~2% variance change: a power check, not a detectable effect."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    shocks = rng.standard_normal(n)
    vol = np.empty(n); vol[0] = 0.012
    for t in range(1, n):
        vol[t] = np.sqrt(0.000002 + 0.08 * (vol[t - 1] * shocks[t - 1]) ** 2 + 0.90 * vol[t - 1] ** 2)
    if vol_signal > 0 and signal == "lunar":
        days = (dates - pd.Timestamp("2000-01-06 18:14")).total_seconds().values / 86400
        vol = vol * np.exp(vol_signal * np.cos(2 * np.pi * (days % 29.530589) / 29.530589))
    elif vol_signal > 0 and signal == "mercury_retro":
        vol = vol * np.exp(vol_signal * _mercury_retrograde(dates))
    elif vol_signal > 0:
        raise ValueError(signal)
    rets = 0.0003 + vol * shocks
    close = 1000 * np.exp(np.cumsum(rets))
    gap = 0.3 * vol * rng.standard_normal(n)
    open_ = np.r_[close[0], close[:-1]] * np.exp(gap)
    span = np.abs(vol * rng.standard_normal(n)) * 0.8
    high = np.maximum(open_, close) * np.exp(span * rng.uniform(0.2, 1.0, n))
    low = np.minimum(open_, close) * np.exp(-span * rng.uniform(0.2, 1.0, n))
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 0.0}, index=dates)
    df.index.name = "date"
    return df
