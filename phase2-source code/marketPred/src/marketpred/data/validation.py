"""Schema and sanity validation. Hard errors raise; soft issues are returned as warnings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import PRICE_COLUMNS, SERIES_COLUMNS


class DataValidationError(ValueError):
    pass


def validate_series(df: pd.DataFrame, name: str = "series") -> list[str]:
    """Validate a long-format series frame. Returns soft warnings; raises on schema violations."""
    missing = set(SERIES_COLUMNS) - set(df.columns)
    if missing:
        raise DataValidationError(f"{name}: missing columns {sorted(missing)}")
    if not isinstance(df["available_at"].dtype, pd.DatetimeTZDtype) or str(df["available_at"].dt.tz) != "UTC":
        raise DataValidationError(f"{name}: available_at must be tz-aware UTC")
    if df["value"].isna().any():
        raise DataValidationError(f"{name}: NaN values must be dropped, not stored")
    if not np.isfinite(df["value"].to_numpy(dtype=float)).all():
        raise DataValidationError(f"{name}: non-finite values")
    dup = df.duplicated(["series_id", "obs_date", "available_at"]).sum()
    if dup:
        raise DataValidationError(f"{name}: {dup} duplicate (series_id, obs_date, available_at) rows")
    # Publication can never precede the start of the reference period.
    early = (df["available_at"].dt.tz_convert(None).dt.normalize() < df["obs_date"]).sum()
    if early:
        raise DataValidationError(f"{name}: {early} rows published before their reference date")
    warnings = []
    for sid, g in df.groupby("series_id"):
        if len(g) < 30:
            warnings.append(f"{sid}: only {len(g)} observations")
    return warnings


def validate_prices(df: pd.DataFrame, name: str = "prices") -> list[str]:
    """Validate an OHLCV frame indexed by session date."""
    missing = set(PRICE_COLUMNS) - set(df.columns)
    if missing:
        raise DataValidationError(f"{name}: missing columns {sorted(missing)}")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise DataValidationError(f"{name}: index must be strictly increasing dates")
    if (df["close"] <= 0).any():
        raise DataValidationError(f"{name}: non-positive close prices")
    warnings = []
    ohlc = df[["open", "high", "low", "close"]].dropna()
    bad = ((ohlc["high"] < ohlc[["open", "close"]].max(axis=1) * (1 - 1e-6))
           | (ohlc["low"] > ohlc[["open", "close"]].min(axis=1) * (1 + 1e-6))).sum()
    if bad:
        warnings.append(f"{name}: {bad} bars with high/low inconsistent with open/close")
    r = np.log(df["close"]).diff().abs()
    jumps = (r > 0.15).sum()
    if jumps:
        warnings.append(f"{name}: {jumps} daily moves > 15% (check splits/errors): {list(r[r > 0.15].index.date)[:5]}")
    flat_open = (df["open"] == df["close"].shift()).mean()
    if flat_open > 0.2:
        warnings.append(f"{name}: {flat_open:.0%} of opens equal previous close (open data likely unreliable)")
    return warnings


def check_calendar_alignment(dates: pd.DatetimeIndex, sessions: pd.DatetimeIndex) -> dict[str, list]:
    """Compare provider dates to the exchange calendar (extra dates and missing sessions)."""
    dates = pd.DatetimeIndex(dates).normalize()
    sessions = sessions[(sessions >= dates.min()) & (sessions <= dates.max())]
    return {
        "not_sessions": list(dates.difference(sessions)),
        "missing_sessions": list(sessions.difference(dates)),
    }
