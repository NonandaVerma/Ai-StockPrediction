"""Common data schemas and publication-time (point-in-time) rules.

Every non-price observation is stored in a long "series" format:

    series_id    str                  e.g. "treasury:10Y"
    obs_date     datetime64[ns]       the period the value refers to (start of month for monthly data)
    value        float64
    available_at datetime64[ns, UTC]  earliest time the value could have been known by a market participant
    source       str
    revised      bool                 True if the stored value may differ from what was first published

`available_at` is what makes as-of joins leakage-free: a feature at decision time T may only use
rows with available_at <= T.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from ..http import HttpClient

SERIES_COLUMNS = ["series_id", "obs_date", "value", "available_at", "source", "revised"]
PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


def period_end(obs_dates: pd.Series | pd.DatetimeIndex, period: str) -> pd.DatetimeIndex:
    """End date of the reference period. D/W: the date itself; M: last calendar day of the month."""
    d = pd.DatetimeIndex(obs_dates).normalize()
    if period in ("D", "W"):
        return d
    if period == "M":
        return d + pd.offsets.MonthEnd(0)
    if period == "Q":
        return d + pd.offsets.QuarterEnd(0)
    raise ValueError(f"unknown period {period!r}")


def compute_available_at(
    obs_dates: pd.Series | pd.DatetimeIndex,
    period: str = "D",
    release_lag_days: int = 0,
    release_time: str = "00:00",
    release_tz: str = "UTC",
) -> pd.DatetimeIndex:
    """available_at = period_end + release_lag_days at release_time (local release_tz), in UTC."""
    hh, mm = (int(x) for x in release_time.split(":"))
    local = period_end(obs_dates, period) + pd.Timedelta(days=release_lag_days, hours=hh, minutes=mm)
    local = local.tz_localize(release_tz, ambiguous="NaT", nonexistent="shift_forward")
    # An ambiguous DST timestamp only occurs at ~1-2am; shift conservatively by one hour.
    if local.isna().any():
        fallback = (pd.DatetimeIndex(period_end(obs_dates, period)) + pd.Timedelta(
            days=release_lag_days, hours=hh + 1, minutes=mm)).tz_localize(release_tz, ambiguous=False,
                                                                          nonexistent="shift_forward")
        local = local.where(~local.isna(), fallback)
    return local.tz_convert("UTC")


def make_series(
    series_id: str,
    obs_date: pd.Series | pd.DatetimeIndex,
    values: Any,
    source: str,
    rule: dict[str, Any],
    revised: bool | None = None,
) -> pd.DataFrame:
    """Build a long-format series frame, applying the source's publication rule."""
    obs = pd.DatetimeIndex(obs_date).normalize()
    df = pd.DataFrame({
        "series_id": series_id,
        "obs_date": obs,
        # Positional (not label-aligned) values: input Series indexes must not re-align the data.
        "value": pd.to_numeric(pd.Series(np.asarray(values, dtype=object)), errors="coerce").to_numpy(dtype=float),
        "available_at": compute_available_at(
            obs,
            period=rule.get("period", "D"),
            release_lag_days=int(rule.get("release_lag_days", 0)),
            release_time=rule.get("release_time", "00:00"),
            release_tz=rule.get("release_tz", "UTC"),
        ),
        "source": source,
        "revised": bool(rule.get("revised", False) if revised is None else revised),
    })
    # Missing values are dropped, never filled: absence must stay visible downstream.
    return df.dropna(subset=["value"]).reset_index(drop=True)


class Connector(ABC):
    """A data provider. `fetch` returns minimally parsed raw tables; `transform` emits long series."""

    name: str = "base"
    requires_env: tuple[str, ...] = ()

    def __init__(self, client: HttpClient, source_cfg: dict[str, Any], market_cfg: Any) -> None:
        self.client = client
        self.cfg = source_cfg
        self.market = market_cfg

    @abstractmethod
    def fetch(self) -> dict[str, pd.DataFrame]:
        """Download raw data. Keys are dataset names; values keep the provider's original columns."""

    @abstractmethod
    def transform(self, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Convert raw tables to the long SERIES_COLUMNS format."""
