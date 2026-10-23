"""Exchange calendars: DST-aware session close timestamps in UTC."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal


@lru_cache(maxsize=8)
def session_schedule(calendar: str, start: str, end: str) -> pd.DataFrame:
    """Schedule indexed by session date with tz-aware UTC `market_open`/`market_close`."""
    return mcal.get_calendar(calendar).schedule(start_date=start, end_date=end)


def decision_times(dates: pd.DatetimeIndex, calendar: str) -> pd.Series:
    """Map exchange session dates to their official close timestamp in UTC.

    A forecast made "at the close of day t" has this decision timestamp: only information
    published at or before it may be used. Early closes and DST are handled by the calendar.
    Dates that are not sessions raise, so calendar mismatches cannot pass silently.
    """
    dates = pd.DatetimeIndex(dates).normalize()
    end = (dates.max() + pd.Timedelta(days=1)).date()
    sched = session_schedule(calendar, str(dates.min().date()), str(end))
    closes = sched["market_close"].copy()
    closes.index = pd.DatetimeIndex(closes.index).normalize()
    missing = dates.difference(closes.index)
    if len(missing):
        raise ValueError(f"{len(missing)} dates are not {calendar} sessions, e.g. {list(missing[:3])}")
    out = closes.reindex(dates)
    out.index.name = "date"
    return out.dt.tz_convert("UTC")


def trading_days(calendar: str, start: str, end: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(session_schedule(calendar, start, end).index).normalize()
