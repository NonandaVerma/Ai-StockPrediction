"""Category F - weather (experimental for an equity index).

Hypothesis tested: mood effects of weather in the city of the exchange (Saunders 1993;
Hirshleifer & Shumway 2003), using NASA POWER reanalysis for New York City. Anomalies are
measured against the mean of *earlier* observations in the same calendar month (causal).
Observed (reanalysis) weather is used with a one-day lag; archived forecasts are not available.
A stronger mechanism exists for agricultural/energy commodities, which are configurable markets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FeatureContext, seasonal_anomaly

CAT = "weather"


def build(ctx: FeatureContext) -> pd.DataFrame:
    f: dict[str, pd.Series] = {}

    def add(name: str, values: pd.Series, source: str, transform: str) -> None:
        f[ctx.register(CAT, name, values, source, transform)] = values

    add("cloud_anom", ctx.aligned("weather:CLOUD_AMT", seasonal_anomaly, "seasonal_anom"),
        "nasa_power:CLOUD_AMT", "minus prior same-month mean")
    add("clearness_anom", ctx.aligned("weather:ALLSKY_KT", seasonal_anomaly, "seasonal_anom"),
        "nasa_power:ALLSKY_KT", "minus prior same-month mean")
    add("temp_anom", ctx.aligned("weather:T2M", seasonal_anomaly, "seasonal_anom"),
        "nasa_power:T2M", "minus prior same-month mean")
    add("log_precip", ctx.aligned("weather:PRECTOTCORR", lambda s: np.log1p(s.clip(lower=0)), "log1p"),
        "nasa_power:PRECTOTCORR", "log1p")
    add("cloud_anom_ma5", ctx.aligned("weather:CLOUD_AMT", lambda s: seasonal_anomaly(s).rolling(5).mean(),
                                      "seasonal_anom_ma5"), "nasa_power:CLOUD_AMT", "5-day mean anomaly")
    return pd.DataFrame(f, index=ctx.dates)
