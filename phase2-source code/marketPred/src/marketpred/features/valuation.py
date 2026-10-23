"""Category C - fundamentals. For an index, aggregate valuation ratios (Campbell & Shiller 1988;
Fama & French 1988). Earnings/dividends come from Shiller's monthly data with a 120-day publication lag;
prices are our own point-in-time close, so ratios never use the monthly-average price.

Single-stock fundamentals (filings, surprises, estimate revisions) need licensed point-in-time
databases (e.g. Compustat/I/B/E/S) and are not implemented; see docs/DATA_SOURCES.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FeatureContext

CAT = "valuation"


def build(ctx: FeatureContext) -> pd.DataFrame:
    p = np.log(ctx.prices["close"].reindex(ctx.dates))
    e = ctx.aligned("shiller:E")
    d = ctx.aligned("shiller:D")
    e10 = ctx.aligned("shiller:E", lambda s: s.rolling(120, min_periods=120).mean(), "ma120")
    f = {
        ctx.register(CAT, "log_ep", None, "shiller:E", "log(E/P_t)"): np.log(e) - p,
        ctx.register(CAT, "log_dp", None, "shiller:D", "log(D/P_t)"): np.log(d) - p,
        ctx.register(CAT, "log_ep10", None, "shiller:E", "log(mean10y E / P_t)"): np.log(e10) - p,
    }
    return pd.DataFrame(f, index=ctx.dates)
