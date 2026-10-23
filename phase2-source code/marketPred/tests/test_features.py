"""Feature calculations, causality and astronomical conventions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from marketpred.features import astro, calendar_features, price
from marketpred.features.base import seasonal_anomaly
from marketpred.features.news import ma_ratio


def test_price_features_manual_values(prices):
    f = price.compute(prices)
    c = prices["close"]
    i = 300
    assert np.isclose(f["ret_5"].iloc[i], np.log(c.iloc[i] / c.iloc[i - 5]))
    r = np.log(c).diff()
    assert np.isclose(f["rv_21"].iloc[i], r.iloc[i - 20:i + 1].std() * np.sqrt(252))
    assert np.isclose(f["ma_ratio_50"].iloc[i], np.log(c.iloc[i] / c.iloc[i - 49:i + 1].mean()))
    assert np.isclose(f["dist_high_252"].iloc[i], np.log(c.iloc[i] / c.iloc[i - 251:i + 1].max()))
    assert np.isclose(f["mom_12_1"].iloc[i], f["ret_252"].iloc[i] - f["ret_21"].iloc[i])
    assert 0 <= f["rsi_14"].dropna().min() and f["rsi_14"].dropna().max() <= 100


def test_price_features_are_causal(prices):
    """Perturbing data after date t must leave every feature at or before t unchanged."""
    base = price.compute(prices)
    cut = 500
    p2 = prices.copy()
    p2.iloc[cut + 1:, :] = p2.iloc[cut + 1:, :] * 3.0
    pert = price.compute(p2)
    pd.testing.assert_frame_equal(base.iloc[:cut + 1], pert.iloc[:cut + 1])


def test_seasonal_anomaly_uses_only_prior_observations():
    idx = pd.date_range("2000-01-01", periods=800, freq="D")
    s = pd.Series(np.random.default_rng(1).normal(size=800), index=idx)
    a = seasonal_anomaly(s)
    s2 = s.copy()
    s2.iloc[400:] += 100
    a2 = seasonal_anomaly(s2)
    assert np.allclose(a.iloc[:400].dropna(), a2.iloc[:400].dropna())


def test_news_ratio_is_causal():
    s = pd.Series(np.arange(1, 801, dtype=float), index=pd.date_range("2000-01-01", periods=800))
    f = ma_ratio(7, 365)
    s2 = s.copy()
    s2.iloc[600:] = 1e6
    assert np.allclose(f(s).iloc[:600].dropna(), f(s2).iloc[:600].dropna())


def test_calendar_features(sessions):
    d = sessions[(sessions >= "2019-11-25") & (sessions <= "2019-12-03")]
    f = calendar_features.compute(d, sessions)
    assert f.loc["2019-11-27", "pre_holiday"] == 1.0      # day before Thanksgiving
    assert f.loc["2019-11-29", "post_holiday"] == 1.0
    assert f.loc["2019-11-29", "turn_of_month"] == 1.0    # last session of November
    assert f.loc["2019-12-03", "turn_of_month"] == 1.0    # 2nd session of December
    assert f.loc["2019-11-26", "turn_of_month"] == 0.0
    assert (f[[f"dow_{i}" for i in range(5)]].sum(axis=1) == 1).all()


def test_astro_known_events():
    t = pd.DatetimeIndex(["2024-04-08 20:00", "2024-04-23 20:00", "2023-12-20 20:00", "2024-02-01 20:00"],
                         tz="UTC")
    f = astro.compute(t)
    # 2024-04-08: total solar eclipse at new moon, within the Mercury retrograde of Apr 1-25 2024
    assert f["eclipse_window"].iloc[0] == 1 and f["near_new_moon"].iloc[0] == 1 and f["moon_illum"].iloc[0] < 0.02
    assert f["mercury_retrograde"].iloc[0] == 1
    # 2024-04-23: full moon
    assert f["near_full_moon"].iloc[1] == 1 and f["moon_illum"].iloc[1] > 0.98
    # Mercury retrograde Dec 13 2023 - Jan 1 2024; direct on Feb 1 2024
    assert f["mercury_retrograde"].iloc[2] == 1 and f["mercury_retrograde"].iloc[3] == 0


def test_eclipse_catalogue_matches_published_central_eclipses():
    found = {(t.date().isoformat(), k) for t, k in astro.eclipse_dates(2017, 2025)}
    published = {("2017-08-21", "solar"), ("2019-07-02", "solar"), ("2020-12-14", "solar"),
                 ("2023-10-14", "solar"), ("2024-04-08", "solar"), ("2018-01-31", "lunar"),
                 ("2019-01-21", "lunar"), ("2022-11-08", "lunar"), ("2025-03-14", "lunar"),
                 ("2025-09-07", "lunar")}
    assert published <= found
