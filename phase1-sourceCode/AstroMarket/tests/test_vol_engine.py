"""Volatility target for Model Comparison (src/vol_engine.py). Offline."""
import numpy as np
import pandas as pd

from src.vol_engine import har_features, realized_vol_target


def test_vol_target_covers_exactly_the_window_after_t():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 200))
    y = realized_vol_target(r, gap=2, horizon=5)
    t = 50
    assert np.isclose(y.iloc[t], np.sqrt(252 * np.mean(r.iloc[t + 3:t + 8] ** 2)))   # sessions t+3..t+7
    assert y.iloc[-7:].isna().all()
    r2 = r.copy(); r2.iloc[t + 8:] *= 10                       # after the window: label unchanged
    assert np.isclose(realized_vol_target(r2, 2, 5).iloc[t], y.iloc[t])


def test_har_features_are_point_in_time():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 400))
    r2 = r.copy(); r2.iloc[300:] *= 5
    pd.testing.assert_frame_equal(har_features(r).iloc[:300], har_features(r2).iloc[:300])


def test_vol_compare_runs_and_har_beats_naive(tmp_path):
    from tests.test_forecast import service
    from src.ephemeris import AstroSettings
    from src.forecast_engine import build_dataset, sky_for_sessions
    from src.vol_engine import vol_compare
    ms = service(tmp_path, start="2010-01-01", end="2018-12-31")
    px = ms.daily("^NSEI"); cal = ms.calendar("^NSEI", px.index)
    sky = sky_for_sessions(px.index.append(cal.next_sessions(px.index[-1], 1)), cal, AstroSettings())
    ds = build_dataset(px, sky, 0, 20, None, None, {"exchange": "NSE", "symbol": "^NSEI", "fetch": False})
    out = vol_compare(ds, ["market", "c02", "c19"], folds=4)
    assert out["status"] == "ok"
    t = out["table"].set_index("model")
    assert t.loc["HAR (ridge) - base", "qlike"] < t.loc["Naive: trailing 20-day vol", "qlike"]   # GARCH data
    assert {"HAR + 02 Technical indicators & trend", "HAR + all selected groups"} <= set(t.index)
    assert t["dm_p_holm"].notna().sum() == 4
