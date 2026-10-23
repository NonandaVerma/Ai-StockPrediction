"""Category B - macroeconomics, rates and financial conditions.

Mechanisms: term spread and short rates as business-cycle / discount-rate proxies (Fama & French 1989;
Estrella & Mishkin 1998); financial conditions (Chicago Fed NFCI); inflation and labour-market slack;
consumer sentiment. Monthly data enter only after conservative publication lags (configs/default.yaml).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FeatureContext, diff, log_diff

CAT = "macro"


def build(ctx: FeatureContext) -> pd.DataFrame:
    f: dict[str, pd.Series] = {}

    def add(name: str, values: pd.Series, source: str, transform: str) -> None:
        f[ctx.register(CAT, name, values, source, transform)] = values

    y3m, y2, y10 = (ctx.aligned(f"treasury:{t}") for t in ("3M", "2Y", "10Y"))
    add("y3m", y3m, "treasury:3M", "level")
    add("y10", y10, "treasury:10Y", "level")
    add("slope_10y_3m", y10 - y3m, "treasury", "10Y-3M")
    add("slope_10y_2y", y10 - y2, "treasury", "10Y-2Y")
    add("d21_y10", ctx.aligned("treasury:10Y", diff(21), "diff21"), "treasury:10Y", "21-obs change")
    add("d63_y3m", ctx.aligned("treasury:3M", diff(63), "diff63"), "treasury:3M", "63-obs change")
    add("nfci", ctx.aligned("nfci:NFCI"), "nfci:NFCI", "level (latest vintage)")
    add("anfci", ctx.aligned("nfci:ANFCI"), "nfci:ANFCI", "level (latest vintage)")
    add("d4_nfci", ctx.aligned("nfci:NFCI", diff(4), "diff4"), "nfci:NFCI", "4-week change")
    add("cpi_yoy", ctx.aligned("bls:CPI_NSA", log_diff(12), "logdiff12"), "bls:CPI_NSA", "12m log change")
    add("cpi_yoy_chg3", ctx.aligned("bls:CPI_NSA", lambda s: np.log(s).diff(12).diff(3), "yoy_diff3"),
        "bls:CPI_NSA", "3m change of yoy")
    add("unrate", ctx.aligned("bls:UNRATE"), "bls:UNRATE", "level")
    add("unrate_gap12", ctx.aligned("bls:UNRATE", lambda s: s - s.rolling(12).min(), "minus_min12"),
        "bls:UNRATE", "level minus 12m min")
    add("umich_ics", ctx.aligned("umich:ICS"), "umich:ICS", "level")
    add("umich_chg3", ctx.aligned("umich:ICS", diff(3), "diff3"), "umich:ICS", "3m change")
    return pd.DataFrame(f, index=ctx.dates)
