"""Category D - cross-market and derivatives information.

VIX level/changes and term structure, the variance risk premium proxy VIX^2 - RV^2
(Bollerslev, Tauchen & Zhou 2009), dollar, gold, oil, copper/gold, high-yield vs investment-grade
credit ETFs (credit risk appetite, cf. Gilchrist & Zakrajsek 2012) and small-vs-large relative strength.
Instruments that settle after the 16:00 ET equity close (VIX 16:15, futures/DXY ~17:00) are only
usable from the next decision time, which the as-of join enforces automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import price
from .base import FeatureContext, log_diff

CAT = "cross_asset"


def build(ctx: FeatureContext) -> pd.DataFrame:
    f: dict[str, pd.Series] = {}

    def add(name: str, values: pd.Series, source: str, transform: str) -> None:
        f[ctx.register(CAT, name, values, source, transform)] = values

    vol_idx = ctx.cfg.market.vol_index
    vix = ctx.aligned(f"{vol_idx}:close")
    rv21 = price.compute(ctx.prices)["rv_21"].reindex(ctx.dates)
    # Log level: volatility targets are modelled in logs, and log VIX is closer to Gaussian.
    add("log_vix", np.log(vix), vol_idx, "log level")
    add("vix_chg5", ctx.aligned(f"{vol_idx}:close", log_diff(5), "logdiff5"), vol_idx, "5-obs log change")
    if vol_idx != "^VIX":  # non-US market: keep US VIX as a global risk-appetite measure
        add("log_us_vix", np.log(ctx.aligned("^VIX:close")), "^VIX", "log level")
    add("vix_term", ctx.combined("^VIX:close", "^VIX3M:close", lambda a, b: np.log(a / b), "log_ratio"),
        "^VIX,^VIX3M", "log(VIX/VIX3M)")
    add("vrp", (vix / 100) ** 2 - rv21 ** 2, f"{vol_idx},target", "implied^2 - RV21^2")
    add("dxy_ret21", ctx.aligned("DX-Y.NYB:close", log_diff(21), "logdiff21"), "DX-Y.NYB", "21-obs log ret")
    add("gold_ret21", ctx.aligned("GC=F:close", log_diff(21), "logdiff21"), "GC=F", "21-obs log ret")
    add("oil_ret21", ctx.aligned("CL=F:close", log_diff(21), "logdiff21"), "CL=F", "21-obs log ret")
    add("copper_gold_chg63", ctx.combined("HG=F:close", "GC=F:close",
                                          lambda a, b: np.log(a / b).diff(63), "logratio_diff63"),
        "HG=F,GC=F", "63-obs change in log(copper/gold)")
    add("credit_hyg_lqd_chg21", ctx.combined("HYG:close", "LQD:close",
                                             lambda a, b: np.log(a / b).diff(21), "logratio_diff21"),
        "HYG,LQD", "21-obs change in log(HYG/LQD)")
    own_ret63 = np.log(ctx.prices["close"]).diff(63).reindex(ctx.dates)
    add("size_rel63", ctx.aligned("^RUT:close", log_diff(63), "logdiff63") - own_ret63, "^RUT,target",
        "63d log ret of Russell 2000 minus target index")
    return pd.DataFrame(f, index=ctx.dates)
