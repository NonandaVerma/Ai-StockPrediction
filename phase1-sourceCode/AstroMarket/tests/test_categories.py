"""Catalog-category feature groups for the dashboard engine (src/catalog_features.py). Offline."""
import numpy as np
import pandas as pd

from src import catalog_features as cf
from src.risk_data import synthetic_ohlc


def _px(seed=1, start="2012-01-01", end="2016-12-31"):
    o = synthetic_ohlc(start, end, seed)
    o["volume"] = 1e6 * np.exp(np.random.default_rng(seed).normal(0, 0.3, len(o)))
    o["price"], o["adj_close"] = o["close"], o["close"]
    o["log_ret"] = np.log(o["price"]).diff()
    return o


def test_own_price_categories_are_point_in_time():
    px = _px()
    px2 = px.copy()
    px2.iloc[-100:, px2.columns.get_indexer(["open", "high", "low", "close", "price", "adj_close"])] *= 3.0
    px2.iloc[-100:, px2.columns.get_loc("volume")] *= 50
    early = px.index[:-100]
    for fn in (cf.price_returns, cf.technical, cf.volume_liquidity, cf.volatility_risk):
        pd.testing.assert_frame_equal(fn(px).loc[early], fn(px2).loc[early], check_exact=False, rtol=1e-9)


def test_split_does_not_create_a_fake_overnight_gap():
    px = _px()
    k = 200                                        # 2:1 split: raw prices halve, adjusted prices do not
    raw = ["open", "high", "low", "close"]
    px.iloc[:k, px.columns.get_indexer(raw)] *= 2.0
    f = cf.price_returns(px)
    assert abs(f["overnight_return_1d"].iloc[k]) < 0.1


def test_rsi_edge_cases():
    idx = pd.bdate_range("2020-01-01", periods=30)
    up = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": np.linspace(100, 130, 30)}, index=idx)
    up["price"] = up["close"]
    assert (cf.technical(up)["rsi_2d"].dropna() == 100).all()
    flat = up.assign(close=100.0, price=100.0)
    assert (cf.technical(flat)["rsi_2d"].dropna() == 50).all()


def test_karana_and_vedic_ids():
    sky = pd.DataFrame({"sun_lon": [0.0] * 5, "moon_lon": [3.0, 9.0, 345.0, 351.0, 357.0],
                        "elongation": [3.0, 9.0, 345.0, 351.0, 357.0]})
    v = cf.astro_vedic(sky)
    # half-tithi 0 = Kimstughna (10), 1 = first movable (0), 57/58/59 = Shakuni/Chatushpada/Naga (7/8/9)
    assert v["karana_id"].tolist() == [10, 0, 7, 8, 9]
    assert v["tithi_id"].tolist() == [1, 1, 29, 30, 30]
    assert v["new_moon_session_flag"].tolist() == [1, 1, 0, 1, 1]


def test_holiday_flags_ignore_weekends():
    sessions = pd.bdate_range("2024-01-01", "2024-01-31").drop(pd.Timestamp("2024-01-17"))   # a Wednesday closure
    tgt = pd.Series(sessions, index=sessions)
    f = cf.calendar(tgt, sessions, "NSE")
    assert f.loc["2024-01-16", "pre_holiday_session_flag"] == 1 and f.loc["2024-01-18", "first_session_after_holiday"] == 1
    assert f.loc["2024-01-05", "pre_holiday_session_flag"] == 0          # Friday before a normal weekend
    assert f.loc["2024-01-31", "month_end_days"] == 0 and f.loc["2024-01-30", "month_end_days"] == 1


def test_category_groups_build_offline():
    px = _px()
    sky = pd.DataFrame({f"{b}_lon": np.random.default_rng(0).uniform(0, 360, len(px)) for b in
                        ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "rahu", "ketu"]},
                       index=px.index)
    for b in ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"]:
        sky[f"{b}_speed"], sky[f"{b}_retro"] = 1.0, 0
    sky["elongation"] = (sky["moon_lon"] - sky["sun_lon"]) % 360
    tgt = pd.Series(px.index, index=px.index)
    groups, _ = cf.build_category_groups(px, sky, tgt, px.index, "NSE", "X.NS", fetch=False)
    assert {"c01", "c02", "c03", "c04", "c16", "c17", "c18", "c19", "c21"} <= set(groups)
    # 20 pairs x 1 (angular_separation_deg only): the 5 named orb columns per pair were dropped as
    # redundant (each is a fixed shift of the same separation angle, corr ~1.0 with it in practice),
    # and the sun-moon pair is skipped (duplicates astro_positions' moon_phase_sin/cos).
    assert groups["c17"].shape[1] == 20


def test_category_models_run_in_the_engine(tmp_path):
    from tests.test_forecast import service
    from src.ephemeris import AstroSettings
    from src.forecast_engine import build_dataset, common_dates, compare, sky_for_sessions
    from src.training import TrainSettings
    ms = service(tmp_path, start="2012-01-01", end="2018-12-31")
    px = ms.daily("^NSEI"); cal = ms.calendar("^NSEI", px.index)
    sky = sky_for_sessions(px.index.append(cal.next_sessions(px.index[-1], 1)), cal, AstroSettings())
    ds = build_dataset(px, sky, 0, 5, None, None, {"exchange": "NSE", "symbol": "^NSEI", "fetch": False})
    res = compare(ds, ["market_linear", "market_c02_linear", "market_c17_linear", "all_categories_linear",
                       "market_c10_linear"], TrainSettings(folds=4), 10)
    assert "market_c10_linear" not in res                      # downloads skipped on synthetic data
    idx = common_dates(res)
    assert len(idx) and all(idx.isin(r["oof"].index).all() for r in res.values())   # shared dates exist
