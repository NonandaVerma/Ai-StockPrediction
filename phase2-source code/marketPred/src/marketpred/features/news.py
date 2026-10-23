"""Category E - news, politics and geopolitics, via published news-based indices.

* Daily US Economic Policy Uncertainty (Baker, Bloom & Davis 2016): newspaper-article counts.
* Daily Geopolitical Risk index (Caldara & Iacoviello 2022), with "acts" and "threats" components.

No licensed historical news archive is available, so article-level NLP features cannot be
backtested here. `marketpred.features.news_nlp` implements the event-extraction/deduplication
pipeline for user-supplied archives; it is unit-tested but not part of the evaluated feature set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FeatureContext, log_ma

CAT = "news"


def ma_ratio(short: int, long: int):
    def f(s: pd.Series) -> pd.Series:
        s = s.clip(lower=1e-9)
        return np.log(s.rolling(short, min_periods=max(1, short // 2)).mean()
                      / s.rolling(long, min_periods=long // 2).mean())
    return f


def build(ctx: FeatureContext) -> pd.DataFrame:
    f: dict[str, pd.Series] = {}

    def add(name: str, values: pd.Series, source: str, transform: str) -> None:
        f[ctx.register(CAT, name, values, source, transform)] = values

    add("epu_log_ma7", ctx.aligned("epu:daily", log_ma(7), "logma7"), "epu:daily", "log 7-day mean")
    add("epu_ma7_vs_ma365", ctx.aligned("epu:daily", ma_ratio(7, 365), "ma7_ma365"), "epu:daily",
        "log(ma7/ma365)")
    add("gpr_log_ma7", ctx.aligned("gpr:GPRD", log_ma(7), "logma7"), "gpr:GPRD", "log 7-day mean")
    add("gpr_ma7_vs_ma365", ctx.aligned("gpr:GPRD", ma_ratio(7, 365), "ma7_ma365"), "gpr:GPRD",
        "log(ma7/ma365)")
    add("gpr_act_log_ma7", ctx.aligned("gpr:GPRD_ACT", log_ma(7), "logma7"), "gpr:GPRD_ACT", "log 7-day mean")
    add("gpr_threat_log_ma7", ctx.aligned("gpr:GPRD_THREAT", log_ma(7), "logma7"), "gpr:GPRD_THREAT",
        "log 7-day mean")
    return pd.DataFrame(f, index=ctx.dates)
