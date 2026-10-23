"""Feature-engineering context: point-in-time access to prices and published series."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..config import Config
from ..data import pit

SEP = "__"  # feature names are "<category>__<name>"


@dataclass
class FeatureContext:
    cfg: Config
    prices: pd.DataFrame            # OHLCV indexed by session date (only sessions <= last close)
    decision_ts: pd.Series          # session date -> UTC close timestamp
    series: pd.DataFrame            # long-format point-in-time series
    provenance: dict[str, dict] = field(default_factory=dict)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.decision_ts.index)

    def raw(self, series_id: str) -> pd.DataFrame:
        return self.series[self.series["series_id"] == series_id]

    def source_rule(self, series_id: str) -> dict:
        df = self.raw(series_id)
        if df.empty:
            return {}
        src = df["source"].iloc[0]
        scfg = self.cfg.source(src)
        if src == "yahoo_cross_asset":
            return scfg.get("tickers", {}).get(series_id.split(":")[0], {})
        return scfg

    def staleness(self, series_id: str) -> float | None:
        return self.source_rule(series_id).get("max_staleness_days")

    def aligned(self, series_id: str, func: Callable[[pd.Series], pd.Series] | None = None,
                desc: str = "level") -> pd.Series:
        """Point-in-time value of a (optionally transformed) series at every decision timestamp."""
        df = self.raw(series_id)
        if df.empty:
            return pd.Series(np.nan, index=self.dates)
        if func is not None:
            df = pit.derive(df, func, f"{series_id}|{desc}")
        return align_values(df, self.decision_ts, self.staleness(series_id))

    def combined(self, a: str, b: str, func: Callable[[pd.Series, pd.Series], pd.Series],
                 desc: str) -> pd.Series:
        """Point-in-time transform of two series joined on obs_date (available when both are)."""
        da, db = self.raw(a), self.raw(b)
        if da.empty or db.empty:
            return pd.Series(np.nan, index=self.dates)
        m = da.merge(db, on="obs_date", suffixes=("_a", "_b")).sort_values("obs_date")
        joined = pd.DataFrame({
            "obs_date": m["obs_date"],
            "value": func(m.set_index("obs_date")["value_a"], m.set_index("obs_date")["value_b"]).to_numpy(),
            "available_at": np.maximum(m["available_at_a"], m["available_at_b"]),
            "series_id": f"{a}&{b}|{desc}", "source": m["source_a"], "revised": m["revised_a"] | m["revised_b"],
        })
        joined["available_at"] = joined["available_at"].cummax()
        joined = joined.dropna(subset=["value"])
        return align_values(joined, self.decision_ts, self.staleness(a))

    def register(self, category: str, name: str, values: pd.Series, source: str, transform: str) -> str:
        col = f"{category}{SEP}{name}"
        self.provenance[col] = {"category": category, "source": source, "transform": transform}
        return col


def align_values(df: pd.DataFrame, decision_ts: pd.Series, max_staleness: float | None) -> pd.Series:
    return pit.align(df, decision_ts, max_staleness)["value"].astype(float)


# ---- causal transforms on obs-indexed series --------------------------------------------------
def log_diff(k: int) -> Callable[[pd.Series], pd.Series]:
    return lambda s: np.log(s).diff(k)


def diff(k: int) -> Callable[[pd.Series], pd.Series]:
    return lambda s: s.diff(k)


def rolling_z(window: int, min_periods: int | None = None) -> Callable[[pd.Series], pd.Series]:
    def f(s: pd.Series) -> pd.Series:
        r = s.rolling(window, min_periods=min_periods or window // 2)
        return (s - r.mean()) / r.std()
    return f


def log_ma(window: int) -> Callable[[pd.Series], pd.Series]:
    return lambda s: np.log(s.clip(lower=1e-9).rolling(window, min_periods=max(1, window // 2)).mean())


def seasonal_anomaly(s: pd.Series) -> pd.Series:
    """Deviation from the mean of all *earlier* observations in the same calendar month (causal)."""
    prior_mean = s.groupby(s.index.month).transform(lambda x: x.shift().expanding(min_periods=3).mean())
    return s - prior_mean
