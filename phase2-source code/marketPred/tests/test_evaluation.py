"""Metrics, statistical tests, backtest accounting, NLP pipeline, drift."""

from __future__ import annotations

import numpy as np
import pandas as pd

from marketpred.backtesting.simulate import simulate_long_cash
from marketpred.evaluation.metrics import binary_metrics, expected_calibration_error, qlike_obs
from marketpred.evaluation.stats import block_bootstrap_ci, diebold_mariano, holm_adjust, kupiec_pof
from marketpred.features.news_nlp import daily_event_features, process_articles
from marketpred.monitoring.drift import psi


def test_holm_adjustment():
    assert np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_dm_detects_better_forecast_and_not_equal_ones():
    rng = np.random.default_rng(0)
    y = rng.normal(size=2000)
    good, bad = y + rng.normal(scale=0.5, size=2000), y + rng.normal(scale=1.0, size=2000)
    res = diebold_mariano((y - bad) ** 2, (y - good) ** 2)
    assert res["mean_diff"] > 0 and res["p_value"] < 0.01
    e1, e2 = rng.normal(size=2000), rng.normal(size=2000)
    assert diebold_mariano(e1 ** 2, e2 ** 2)["p_value"] > 0.01


def test_block_bootstrap_ci_contains_truth():
    rng = np.random.default_rng(1)
    d = rng.normal(0.5, 1, 1000)
    ci = block_bootstrap_ci(lambda ix: d[ix].mean(), len(d), 20, 300)
    assert ci["ci_low"] < 0.5 < ci["ci_high"]


def test_binary_metrics_and_calibration():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.1, 0.9, 20000)
    y = (rng.random(20000) < p).astype(float)
    m = binary_metrics(y, p, base_rate=y.mean())
    assert m["ece"] < 0.02 and m["auc"] > 0.7 and m["brier_skill"] > 0
    assert expected_calibration_error(y, np.clip(p + 0.2, 0, 1)) > 0.1


def test_qlike_minimised_at_truth():
    assert qlike_obs(np.array([0.0]), np.array([0.0]))[0] == 0
    assert qlike_obs(np.array([0.0]), np.array([0.3]))[0] > 0


def test_kupiec():
    v = np.zeros(1000, bool)
    v[:50] = True
    assert kupiec_pof(v, 0.05)["p_value"] > 0.5
    v[:150] = True
    assert kupiec_pof(v, 0.05)["p_value"] < 0.01


def test_backtest_accounting():
    idx = pd.bdate_range("2020-01-01", periods=4)
    prob = pd.Series([0.6, 0.6, 0.4, 0.6], index=idx)
    ret = pd.Series(np.log([1.01, 1.02, 0.97, 1.01]), index=idx)
    res = simulate_long_cash(prob, ret, pd.Series(0.0, index=idx), cost_bps=10, slippage_bps=0)
    d = res["daily"]
    assert np.allclose(d["position"], [1, 1, 0, 1])
    # entry cost day 1, exit cost day 3, re-entry day 4
    assert np.allclose(d["strategy"], [0.01 - 0.001, 0.02, -0.001, 0.01 - 0.001])
    assert res["n_trades"] == 3


def test_news_pipeline_dedup_events_and_point_in_time():
    arts = pd.DataFrame({
        "source": ["A", "B", "C", "D"],
        "published": ["2024-03-04 15:00", "2024-03-04 15:30", "2024-03-04 17:00", "2024-03-05 10:00"],
        "tz": ["America/New_York"] * 4,
        "headline": ["Fed signals rate cut as inflation eases - Reuters",
                     "Fed signals rate cut as inflation eases",
                     "Oil prices surge after shipping blockade in key canal",
                     "Company beats earnings guidance, shares rally"],
    })
    p = process_articles(arts)
    assert p["is_duplicate"].tolist() == [False, True, False, False]
    assert "monetary_policy" in p.loc[0, "event_types"] and "supply_chain" in p.loc[2, "event_types"]
    assert p.loc[3, "sentiment"] > 0
    dec = pd.Series(pd.DatetimeIndex(["2024-03-04 21:00", "2024-03-05 21:00"], tz="UTC"),
                    index=pd.DatetimeIndex(["2024-03-04", "2024-03-05"]))
    daily = daily_event_features(p, dec)
    # 17:00 ET = 22:00 UTC, after the Mar-4 close -> counted on Mar 5; duplicate excluded
    assert daily["n_articles"].tolist() == [1, 2]


def test_psi_detects_shift():
    rng = np.random.default_rng(4)
    a = pd.Series(rng.normal(size=5000))
    assert psi(a, pd.Series(rng.normal(size=500))) < 0.1
    assert psi(a, pd.Series(rng.normal(2, 1, size=500))) > 0.25
