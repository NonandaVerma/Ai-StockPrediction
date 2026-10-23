"""Seasonal and calendar controls (required control set for the astrology experiment).

Documented effects investigated: day-of-week, turn-of-the-month (Ariel 1987; Lakonishok & Smidt 1988),
pre-holiday (Lakonishok & Smidt 1988), January, and the Halloween / "Sell in May" effect
(Bouman & Jacobsen 2002). The exchange holiday schedule is published in advance, so using the
next session date is not look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..calendar import trading_days
from .base import FeatureContext

CAT = "calendar"


def compute(dates: pd.DatetimeIndex, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    dates = pd.DatetimeIndex(dates)
    pos = sessions.get_indexer(dates)
    if (pos < 0).any():
        raise ValueError("calendar features: dates not in session list")
    nxt = sessions[np.minimum(pos + 1, len(sessions) - 1)]
    prv = sessions[np.maximum(pos - 1, 0)]
    f = pd.DataFrame(index=dates)
    for d in range(5):
        f[f"dow_{d}"] = (dates.dayofweek == d).astype(float)
    f["month_sin"] = np.sin(2 * np.pi * dates.month / 12)
    f["month_cos"] = np.cos(2 * np.pi * dates.month / 12)
    f["january"] = (dates.month == 1).astype(float)
    f["nov_apr"] = dates.month.isin([11, 12, 1, 2, 3, 4]).astype(float)
    s_month = pd.Series(sessions.to_period("M"), index=sessions)
    rank_in_month = s_month.groupby(s_month).cumcount().reindex(dates).to_numpy()
    remaining = (s_month.groupby(s_month).cumcount(ascending=False)).reindex(dates).to_numpy()
    f["turn_of_month"] = ((remaining == 0) | (rank_in_month < 3)).astype(float)
    f["days_to_month_end"] = remaining.astype(float)
    bday_gap_next = np.busday_count(dates.values.astype("datetime64[D]"), nxt.values.astype("datetime64[D]"))
    bday_gap_prev = np.busday_count(prv.values.astype("datetime64[D]"), dates.values.astype("datetime64[D]"))
    f["pre_holiday"] = (bday_gap_next > 1).astype(float)
    f["post_holiday"] = (bday_gap_prev > 1).astype(float)
    return f


def build(ctx: FeatureContext) -> pd.DataFrame:
    d = ctx.dates
    sessions = trading_days(ctx.cfg.market.calendar, str((d.min() - pd.Timedelta(days=10)).date()),
                            str((d.max() + pd.Timedelta(days=30)).date()))
    feats = compute(d, sessions)
    feats.columns = [ctx.register(CAT, c, feats[c], "exchange calendar", "deterministic") for c in feats.columns]
    return feats
