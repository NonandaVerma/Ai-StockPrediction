"""Point-in-time (as-of) alignment of published series to decision timestamps.

Rules enforced here:
* A value is usable at decision time T only if its `available_at <= T`.
* A later publication about an *older* period (a backfill or revision of history) never replaces
  the newest observation already known; revisions of the newest observation do replace it.
* Derived series (changes, moving averages, ...) inherit the latest `available_at` of every input
  they depend on. When a series has several vintages, derivations are recomputed from the
  snapshot that was actually available at each publication time.
* Values older than `max_staleness_days` relative to the decision date become missing rather than
  being forward-filled indefinitely.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def has_vintages(df: pd.DataFrame) -> bool:
    return bool(df.duplicated("obs_date").any())


def snapshot(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Values of one series as they were known at `as_of` (latest vintage per obs_date)."""
    known = df[df["available_at"] <= as_of].sort_values(["obs_date", "available_at"])
    known = known.drop_duplicates("obs_date", keep="last")
    return known.set_index("obs_date")["value"]


def derive(df: pd.DataFrame, func: Callable[[pd.Series], pd.Series], series_id: str) -> pd.DataFrame:
    """Apply `func` (obs-indexed Series -> obs-indexed Series, backward-looking) point-in-time.

    `func` must be causal in obs_date order (e.g. diff, rolling mean, pct_change).
    """
    if df.empty:
        return df.assign(series_id=series_id)
    meta = {"source": df["source"].iloc[0], "revised": bool(df["revised"].any())}
    if not has_vintages(df):
        d = df.sort_values("obs_date")
        s = d.set_index("obs_date")["value"]
        out = func(s)
        avail = d["available_at"].cummax().to_numpy()
        res = pd.DataFrame({"obs_date": s.index, "value": out.to_numpy(), "available_at": avail})
    else:
        rows = []
        for a in np.sort(df["available_at"].unique()):
            a = pd.Timestamp(a)
            snap = snapshot(df, a)
            val = func(snap)
            rows.append((snap.index[-1], val.iloc[-1], a))
        res = pd.DataFrame(rows, columns=["obs_date", "value", "available_at"])
    res["available_at"] = pd.to_datetime(res["available_at"], utc=True)
    res = res.dropna(subset=["value"])
    res = res[np.isfinite(res["value"].to_numpy(dtype=float))]
    return res.assign(series_id=series_id, **meta).reset_index(drop=True)


def align(df: pd.DataFrame, decision_ts: pd.Series, max_staleness_days: float | None = None) -> pd.DataFrame:
    """As-of join of one series onto decision timestamps.

    Parameters
    ----------
    df : long-format frame of a single series (obs_date, value, available_at).
    decision_ts : Series indexed by decision date with tz-aware UTC decision timestamps.

    Returns a frame indexed like `decision_ts` with columns value, obs_date, available_at, age_days.
    """
    out_index = decision_ts.index
    if df.empty:
        return pd.DataFrame(index=out_index, columns=["value", "obs_date", "available_at", "age_days"], dtype=float)
    d = df.sort_values(["available_at", "obs_date"]).reset_index(drop=True)
    # Drop publications about periods older than the newest already-known period (backfills).
    newest_known = d["obs_date"].cummax()
    d = d[d["obs_date"] >= newest_known].copy()
    d["available_at"] = d["available_at"].astype("datetime64[ns, UTC]")
    left = pd.DataFrame({"decision_ts": decision_ts.astype("datetime64[ns, UTC]").to_numpy(),
                         "date": out_index}).sort_values("decision_ts")
    merged = pd.merge_asof(left, d[["available_at", "obs_date", "value"]], left_on="decision_ts",
                           right_on="available_at", direction="backward", allow_exact_matches=True)
    merged = merged.set_index("date").reindex(out_index)
    age = (pd.DatetimeIndex(merged.index) - pd.DatetimeIndex(merged["obs_date"])).days
    merged["age_days"] = np.asarray(age, dtype=float)
    if max_staleness_days is not None:
        stale = merged["age_days"] > max_staleness_days
        merged.loc[stale, "value"] = np.nan
    return merged[["value", "obs_date", "available_at", "age_days"]]
