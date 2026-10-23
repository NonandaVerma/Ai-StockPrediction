"""Actual vs Predicted page (src/hindcast.py): predictions never see the prices they predict. Offline."""
import numpy as np
import pandas as pd
import pytest

from src.ephemeris import AstroSettings
from src.hindcast import hindcast
from tests.test_forecast import service


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    ms = service(tmp_path_factory.mktemp("h"), start="2012-01-01", end="2018-12-31")
    px = ms.daily("^NSEI")
    return px, ms.calendar("^NSEI", px.index)


def _corrupt(px, from_date):
    px2 = px.copy()
    m = px2.index >= from_date
    for c in ("open", "high", "low", "close", "adj_close", "price"):
        px2.loc[m, c] *= 3.0
    px2["log_ret"] = np.log(px2["price"]).diff()
    return px2


def test_strict_predictions_ignore_every_price_in_the_period(setup):
    px, cal = setup
    window = px.index[-30:-25]
    a = hindcast(px, cal, AstroSettings(), window, "market_linear", "strict")
    b = hindcast(_corrupt(px, window[0]), cal, AstroSettings(), window, "market_linear", "strict")
    cols = ["predicted_open", "predicted_high", "predicted_low", "predicted_close", "range_low", "range_high", "prob_up"]
    pd.testing.assert_frame_equal(a["predicted"][cols], b["predicted"][cols])
    assert (a["predicted"]["as_of"] == px.index[-31]).all()
    assert a["predicted"]["sessions_ahead"].tolist() == [1, 2, 3, 4, 5]


def test_rolling_prediction_for_a_day_ignores_that_day_and_later(setup):
    px, cal = setup
    window = px.index[-30:-26]
    a = hindcast(px, cal, AstroSettings(), window, "market_linear", "rolling")
    b = hindcast(_corrupt(px, window[2]), cal, AstroSettings(), window, "market_linear", "rolling")
    pa, pb = a["predicted"], b["predicted"]
    pd.testing.assert_series_equal(pa["predicted_close"].iloc[:3], pb["predicted_close"].iloc[:3])
    assert not np.isclose(pa["predicted_close"].iloc[3], pb["predicted_close"].iloc[3])   # day 4 sees day 3


def test_actual_side_is_the_raw_market_data(setup):
    px, cal = setup
    window = px.index[-10:-5]
    out = hindcast(px, cal, AstroSettings(), window, "naive_mean", "strict")
    pd.testing.assert_frame_equal(out["actual"], px.loc[window])
    assert set(out["summary"]) >= {"mae_pct", "direction_hit_rate", "range_coverage"}


def test_rejects_too_long_or_empty_periods(setup):
    px, cal = setup
    with pytest.raises(ValueError):
        hindcast(px, cal, AstroSettings(), px.index[-70:], "naive_mean")
    with pytest.raises(ValueError):
        hindcast(px, cal, AstroSettings(), px.index[:0], "naive_mean")


def test_custom_category_and_algorithm_choice(setup):
    from src.forecast_models import SPEC_BY_NAME, custom_model_name
    px, cal = setup
    window = px.index[-30:-27]
    name = custom_model_name(["weather", "astro", "c19", "astro"], "ensemble")
    assert name == "custom:ensemble:astro+weather+c19" and name not in list(SPEC_BY_NAME)   # built on demand
    spec = SPEC_BY_NAME[name]
    assert spec.groups == ("astro", "weather", "c19") and spec.algo == "ensemble"
    # same data + algorithm as a fixed model gives the same predictions
    a = hindcast(px, cal, AstroSettings(), window, custom_model_name(["market"], "linear"), "strict")["predicted"]
    b = hindcast(px, cal, AstroSettings(), window, "market_linear", "strict")["predicted"]
    pd.testing.assert_series_equal(a["predicted_close"], b["predicted_close"])
    ctx = {"exchange": "NSE", "symbol": "^NSEI", "fetch": False}
    out = hindcast(px, cal, AstroSettings(), window, custom_model_name(["astro", "c19"], "lgbm"), "strict",
                   external_fn=lambda s: ({}, []), category_ctx=ctx)
    assert len(out["predicted"]) == 3
    with pytest.raises(ValueError, match="Weather"):           # no weather data offline: reported, not filled in
        hindcast(px, cal, AstroSettings(), window, name, "strict", external_fn=lambda s: ({}, []), category_ctx=ctx)
    with pytest.raises(ValueError):
        custom_model_name([], "linear")


def test_predicted_candles_are_well_formed(setup):
    px, cal = setup
    window = px.index[-30:-25]
    pr = hindcast(px, cal, AstroSettings(), window, "market_linear", "strict")["predicted"]
    assert (pr["predicted_high"] >= pr[["predicted_open", "predicted_close"]].max(axis=1)).all()
    assert (pr["predicted_low"] <= pr[["predicted_open", "predicted_close"]].min(axis=1)).all()
    assert pr["predicted_open"].iloc[0] == pytest.approx(px["close"].loc[:window[0]].iloc[-2])  # last close before
    assert np.allclose(pr["predicted_open"].iloc[1:].to_numpy(), pr["predicted_close"].iloc[:-1].to_numpy())
