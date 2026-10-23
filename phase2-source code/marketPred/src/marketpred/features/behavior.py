"""Category G - investor behaviour and positioning: CFTC Commitments of Traders (E-mini S&P 500).

Net positions of non-commercial (speculative), commercial and non-reportable (small) traders scaled
by open interest. Evidence on the predictive value of COT positioning is mixed (e.g. Sanders, Boris &
Manfredo 2004 for energy futures); treated as a hypothesis. Survey sentiment (AAII) blocks automated
downloads (HTTP 403) and is not included; consumer sentiment is in the macro category.
"""

from __future__ import annotations

import pandas as pd

from .base import FeatureContext, diff, log_diff, rolling_z

CAT = "behavior"


def build(ctx: FeatureContext) -> pd.DataFrame:
    f: dict[str, pd.Series] = {}

    def add(name: str, values: pd.Series, source: str, transform: str) -> None:
        f[ctx.register(CAT, name, values, source, transform)] = values

    add("spec_net_oi", ctx.aligned("cftc:noncomm_net_oi"), "cftc", "noncommercial net / OI")
    add("spec_net_oi_chg4", ctx.aligned("cftc:noncomm_net_oi", diff(4), "diff4"), "cftc", "4-week change")
    add("spec_net_oi_z52", ctx.aligned("cftc:noncomm_net_oi", rolling_z(52), "z52"), "cftc", "52-week z-score")
    add("small_net_oi", ctx.aligned("cftc:nonrept_net_oi"), "cftc", "non-reportable net / OI")
    add("oi_chg13", ctx.aligned("cftc:open_interest", log_diff(13), "logdiff13"), "cftc",
        "13-week log change in open interest")
    return pd.DataFrame(f, index=ctx.dates)
