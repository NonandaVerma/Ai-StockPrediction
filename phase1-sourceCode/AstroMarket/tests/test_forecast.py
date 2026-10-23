"""Tests for the Market Prediction dashboard modules. Run with:  pytest -q
All tests are offline (synthetic prices, temporary SQLite database)."""
import numpy as np
import pandas as pd
import pytest

from src import repository as repo
from src.analysis import forward_returns
from src.astrology import chart, detect_events, sky_table
from src.config import load_config
from src.db import connect, migrate
from src.ephemeris import AstroSettings, positions
from src.exchanges import EXCHANGES, TradingCalendar, session_time_utc
from src.forecast_engine import build_dataset, resolve_window, sky_for_sessions
from src.forecast_service import run_forecast
from src.market_service import MarketService, freshness_label
from src.training import TrainSettings, walk_forward


def make_cfg(tmp_path, signal=0.0, start="2008-01-01", end="2020-12-31"):
    return load_config("config.yaml", **{"market.synthetic": True, "market.synthetic_signal": signal,
                                         "forecast.history_start": start, "market.end": end,
                                         "database.path": str(tmp_path / "t.db"),
                                         "forecast.folds": 4, "output_dir": "outputs/_test"})


class CappedSynthetic:
    """Synthetic provider that stops at a fixed end date (keeps tests deterministic)."""
    def __new__(cls, cfg, end):
        from src.market_providers import SyntheticProvider
        p = SyntheticProvider(seed=cfg["seed"], signal=cfg["market"]["synthetic_signal"])
        orig = p.history
        p.history = lambda symbol, start, _end, interval="1d": orig(symbol, start, end, interval)
        return p


def service(tmp_path, signal=0.0, end="2020-12-31", start="2008-01-01"):
    cfg = make_cfg(tmp_path, signal, start, end)
    return MarketService(cfg, provider=CappedSynthetic(cfg, end))


# ---------------------------------------------------------------- database
def test_migrations_idempotent_and_unique(tmp_path):
    con = connect(tmp_path / "a.db")
    assert migrate(con) == []                                   # second run applies nothing
    iid = repo.upsert_instrument(con, "^NSEI", "NSE", "Nifty 50")
    sid = repo.source_id(con, "test")
    bars = pd.DataFrame({"ts_utc": pd.to_datetime(["2024-01-01 10:00", "2024-01-02 10:00"], utc=True),
                         "open": 1.0, "high": 1.0, "low": 1.0, "close": [1.0, 2.0], "adj_close": None,
                         "volume": 0.0}, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    repo.save_bars(con, iid, sid, "1d", bars)
    repo.save_bars(con, iid, sid, "1d", bars.assign(close=[1.5, 2.5]))   # re-import updates, no duplicates
    got = repo.load_bars(con, iid, "1d")
    assert len(got) == 2 and got.close.tolist() == [1.5, 2.5]
    mid = repo.register_model(con, "m", "v1", "astro", {})
    assert repo.register_model(con, "m", "v1", "astro", {}) == mid
    row = {"instrument_id": iid, "model_version_id": mid, "as_of_session": "2024-01-02",
           "window_start": "2024-01-03", "window_end": "2024-01-03", "gap_sessions": 0, "horizon_sessions": 1,
           "zodiac": "sidereal", "ayanamsa": "lahiri", "evidence": "experimental", "generated_at": repo.now_iso()}
    assert repo.save_prediction(con, row) is True
    assert repo.save_prediction(con, {**row, "generated_at": repo.now_iso()}) is False


# ---------------------------------------------------------------- exchanges / calendar
def test_session_times_handle_dst():
    ny = session_time_utc(EXCHANGES["NYSE"], ["2024-01-10", "2024-07-10"], "open")
    assert [t.hour for t in ny] == [14, 13]                     # EST vs EDT
    ldn = session_time_utc(EXCHANGES["LSE"], ["2024-01-10", "2024-07-10"], "open")
    assert [t.hour for t in ldn] == [8, 7]                      # GMT vs BST
    nse = session_time_utc(EXCHANGES["NSE"], ["2024-01-10", "2024-07-10"], "open")
    assert all((t.hour, t.minute) == (3, 45) for t in nse)      # IST has no DST


def test_calendar_maps_weekends_and_holidays():
    hol = pd.DataFrame({"date": pd.to_datetime(["2025-01-27"]), "name": ["Test holiday"]})
    cal = TradingCalendar(EXCHANGES["NSE"], pd.bdate_range("2024-12-02", "2025-01-17"), hol)
    m = cal.map_date("2025-01-18")                              # Saturday
    assert m["session"] == pd.Timestamp("2025-01-20") and "weekend" in m["note"]
    m = cal.map_date("2025-01-27")
    assert m["session"] == pd.Timestamp("2025-01-28") and "Test holiday" in m["note"]
    nxt = cal.next_sessions("2025-01-24", 2)                    # Fri -> Mon is a holiday
    assert list(nxt) == [pd.Timestamp("2025-01-28"), pd.Timestamp("2025-01-29")]
    assert cal.coverage_notes(nxt) == []                        # 2025 holiday list present
    assert cal.coverage_notes(pd.DatetimeIndex(["2026-03-02"]))  # 2026 not covered -> warning


def test_resolve_window_gap_and_horizon():
    cal = TradingCalendar(EXCHANGES["NSE"], pd.bdate_range("2024-01-01", "2024-03-29"),
                          pd.DataFrame(columns=["date", "name"]))
    r = resolve_window(cal, "2024-03-29", "next", 5)
    assert r["gap"] == 0 and r["horizon"] == 5 and r["window"][0] == pd.Timestamp("2024-04-01")
    r = resolve_window(cal, "2024-03-29", "date", date="2024-04-06")   # Saturday -> Monday 8th
    assert r["window"][0] == pd.Timestamp("2024-04-08") and r["gap"] == 5 and r["horizon"] == 1
    r = resolve_window(cal, "2024-03-29", "range", start="2024-04-03", end="2024-04-09")
    assert r["gap"] == 2 and r["horizon"] == 5


# ---------------------------------------------------------------- astronomy / astrology
def test_outer_planets_and_nodes():
    s = AstroSettings(zodiac="tropical")
    p = positions(pd.Timestamp("2024-04-08 18:00", tz="UTC"), s).set_index("body")
    assert 50 < p.loc["uranus", "longitude"] < 52                # Uranus ~21 Taurus
    assert 357 < p.loc["neptune", "longitude"] < 360             # Neptune ~28 Pisces
    assert 300 < p.loc["pluto", "longitude"] < 303               # Pluto ~2 Aquarius
    assert abs((p.loc["ketu", "longitude"] - p.loc["rahu", "longitude"]) % 360 - 180) < 1e-6
    sid = positions(pd.Timestamp("2024-04-08 18:00", tz="UTC"), AstroSettings()).set_index("body")
    ayan = (p.loc["sun", "longitude"] - sid.loc["sun", "longitude"]) % 360
    assert 24 < ayan < 24.3                                      # Lahiri ayanamsa in 2024


def test_eclipse_chart_and_events():
    c = chart(pd.Timestamp("2024-04-08 18:18", tz="UTC"), AstroSettings())
    assert c["lunar"]["tithi_name"] == "Amavasya" and c["lunar"]["illumination"] < 0.01
    assert c["positions"].set_index("body").loc["mercury", "retrograde"]
    days = sky_table(pd.date_range("2024-03-25", "2024-04-30", tz="UTC"), AstroSettings())
    ev = detect_events(days, "sidereal")
    st = ev[(ev.event_type == "station_retrograde") & (ev.body1 == "mercury")]
    assert len(st) == 1 and pd.Timestamp("2024-04-01", tz="UTC") <= st.date.iloc[0] <= pd.Timestamp("2024-04-03", tz="UTC")
    assert (ev.event_type == "new_moon").sum() == 1


def test_forward_returns_alignment():
    lr = pd.Series([np.nan, .01, .02, .03, .04, .05])
    f = forward_returns(lr, gap=1, horizon=2)
    assert f.iloc[0] == pytest.approx(.02 + .03)                 # skips session 1, sums 2..3
    assert np.isnan(f.iloc[3]) and np.isnan(f.iloc[5])


# ---------------------------------------------------------------- market service
def test_freshness_labels_are_conservative():
    ex = EXCHANGES["NSE"]
    open_now = pd.Timestamp("2024-04-09 06:00", tz="UTC")        # 11:30 IST Tuesday
    closed = pd.Timestamp("2024-04-09 12:00", tz="UTC")
    assert freshness_label("yahoo", False, "5m", open_now, ex, open_now) == "delayed"
    assert freshness_label("yahoo", False, "5m", open_now, ex, closed) == "end-of-day"
    assert freshness_label("x", True, "5m", open_now - pd.Timedelta(minutes=10), ex, open_now) == "delayed"
    assert freshness_label("x", True, "5m", open_now, ex, open_now) == "real-time"
    assert freshness_label("yahoo", False, "1d", open_now, ex, open_now) == "end-of-day"
    assert freshness_label("synthetic", False, "1d", open_now, ex, open_now) == "synthetic"


# ---------------------------------------------------------------- forecast engine
def test_forecast_uses_only_point_in_time_data(tmp_path):
    ms = service(tmp_path, start="2012-01-01", end="2018-12-31")
    s = AstroSettings.from_cfg(ms.cfg)
    models = ["naive_mean", "market_linear", "astro_stat"]
    a = run_forecast(ms, "^NSEI", s, as_of="2017-06-30", mode="next", horizon=5, models=models, save=False)
    # corrupt every price after the as-of date: the forecast must not change
    iid = ms.instrument("^NSEI")["id"]
    ms.con.execute("UPDATE price_bars SET close = close * 3, adj_close = adj_close * 3 "
                   "WHERE instrument_id=? AND session_date > '2017-06-30'", (iid,))
    ms.con.commit()
    b = run_forecast(ms, "^NSEI", s, as_of="2017-06-30", mode="next", horizon=5, models=models, save=False)
    for ra, rb in zip(a["models"], b["models"]):
        assert ra["prob_up"] == pytest.approx(rb["prob_up"])
        assert ra["expected_return"] == pytest.approx(rb["expected_return"])
    assert a["as_of"] == pd.Timestamp("2017-06-30") and a["realized"] is not None


def test_walk_forward_respects_embargo(tmp_path):
    ms = service(tmp_path, start="2014-01-01", end="2018-12-31")
    px = ms.daily("^NSEI")
    cal = ms.calendar("^NSEI", px.index)
    sky = sky_for_sessions(px.index.append(cal.next_sessions(px.index[-1], 4)), cal, AstroSettings())
    ds = build_dataset(px, sky, gap=3, horizon=2)
    # Target of row t must start strictly after t (gap) and astro must describe the window start.
    i = 100
    assert ds.target_sessions.iloc[i] == px.index[i + 4]
    assert ds.fwd.iloc[i] == pytest.approx(px.log_ret.iloc[i + 4:i + 6].sum())
    oof = walk_forward(ds, "naive_mean", TrainSettings(folds=4))
    assert oof.index.min() > ds.fwd.dropna().index[0]


def test_insufficient_history_is_reported(tmp_path):
    ms = service(tmp_path, start="2019-06-01", end="2020-06-30")
    out = run_forecast(ms, "^NSEI", AstroSettings(), mode="next", horizon=1,
                       models=["naive_mean", "astro_stat"], save=False)
    assert out["status"] == "unavailable" and "minimum" in out["message"]


@pytest.mark.slow
def test_positive_control_detects_injected_moon_effect(tmp_path):
    """Synthetic prices with an injected lunar-phase effect: Model A must find it and label it validated,
    while the same pipeline on pure noise must not."""
    s = AstroSettings()
    sig = run_forecast(service(tmp_path / "sig", signal=0.004), "^NSEI", s, mode="next", horizon=1,
                       models=["naive_mean", "astro_stat"], save=False)
    noise = run_forecast(service(tmp_path / "noise"), "^NSEI", s, mode="next", horizon=1,
                         models=["naive_mean", "astro_stat"], save=False)
    a_sig = next(r for r in sig["models"] if r["model"] == "astro_stat")
    a_noise = next(r for r in noise["models"] if r["model"] == "astro_stat")
    assert a_sig["evidence"] == "validated" and a_sig["metrics"]["auc"] > 0.55
    assert a_noise["evidence"] != "validated"


def test_help_texts_seeded_and_editable(tmp_path):
    con = connect(tmp_path / "h.db")
    h = repo.help_texts(con)
    assert {"ayanamsa", "auc", "models", "nakshatra"} <= set(h)
    repo.save_help(con, pd.DataFrame([{"key": "AUC", "term": "AUC", "category": "statistics",
                                       "description": "edited"}]))
    assert repo.help_texts(con)["auc"] == "edited"            # keys are normalised to lower-case
    repo.delete_help(con, ["auc"])
    assert "auc" not in repo.help_texts(con)


# ---------------------------------------------------------------- geopolitics / weather models
def _ds_with_external(tmp_path, groups=("geopolitics", "weather")):
    ms = service(tmp_path, start="2012-01-01", end="2018-12-31")
    px = ms.daily("^NSEI")
    cal = ms.calendar("^NSEI", px.index)
    sky = sky_for_sessions(px.index.append(cal.next_sessions(px.index[-1], 1)), cal, AstroSettings())
    rng = np.random.default_rng(0)
    ext = {g: pd.DataFrame(rng.standard_normal((len(px), 3)), index=px.index, columns=[f"{g}_{i}" for i in range(3)])
           for g in groups}
    if "geopolitics" in ext:
        ext["geopolitics"].iloc[:300] = np.nan                       # external data starting later than prices
    return build_dataset(px, sky, 0, 5, ext, ["note"]), px, cal, ms


def test_external_models_skip_missing_groups_and_compare_on_shared_dates(tmp_path):
    from src.forecast_engine import astro_adds_value, common_dates, compare
    ds, *_ = _ds_with_external(tmp_path, groups=("geopolitics",))       # no weather (e.g. a US listing)
    res = compare(ds, ["market_linear", "market_geo_linear", "market_weather_linear"], TrainSettings(folds=4), 10)
    assert set(res) == {"market_linear", "market_geo_linear"}
    a, b = res["market_linear"]["oof"], res["market_geo_linear"]["oof"]
    assert b.index.min() > ds.px.index[300]                              # rows without GPR are not used
    assert a.index.min() < b.index.min()                                 # each model scored on its own dates
    assert common_dates(res).equals(a.index.intersection(b.index))
    vs = astro_adds_value(res, ds.horizon)[0]                            # head-to-head on shared dates only
    assert vs["model"] == "market_geo_linear" and vs["n_independent"] == len(common_dates(res)[::5])


def test_model_scorecard_does_not_depend_on_other_selected_models(tmp_path):
    from src.forecast_engine import compare
    ds, *_ = _ds_with_external(tmp_path, groups=("geopolitics",))
    t = TrainSettings(folds=4)
    alone = compare(ds, ["market_linear"], t, 10)["market_linear"]
    with_geo = compare(ds, ["market_linear", "market_geo_linear"], t, 10)["market_linear"]
    pd.testing.assert_frame_equal(alone["oof"], with_geo["oof"])
    assert alone["metrics"]["auc"] == with_geo["metrics"]["auc"]


def test_forecast_and_actual_vs_predicted_agree(tmp_path):
    """Future Forecast of the next k sessions == Actual vs Predicted (strict) day k from the same as-of."""
    from src.forecast_service import run_hindcast
    from src.training import TrainSettings
    ms = service(tmp_path, start="2012-01-01", end="2018-12-31")
    s = AstroSettings.from_cfg(ms.cfg)
    px = ms.daily("^NSEI")
    as_of, window = px.index[-40], px.index[-39:-36]
    hc = run_hindcast(ms, "^NSEI", s, window, "market_linear", "strict")["predicted"]
    for k in (1, 3):
        fc = run_forecast(ms, "^NSEI", s, as_of=as_of, mode="next", horizon=k, models=["market_linear"], save=False)
        r = fc["models"][0]
        assert r["prob_up"] == pytest.approx(hc["prob_up"].iloc[k - 1])
        assert r["expected_return"] == pytest.approx(hc["expected"].iloc[k - 1])
        assert r["interval_low"] == pytest.approx(hc["lo"].iloc[k - 1])
    assert fc["training"] == TrainSettings.from_cfg(ms.cfg).describe()


def test_training_options_change_the_fit_but_never_use_future_rows(tmp_path):
    from src.training import fit_final, usable_rows
    ds, *_ = _ds_with_external(tmp_path, groups=("weather",))
    full, win = TrainSettings(folds=4), TrainSettings(folds=4, train_window_years=3)
    assert usable_rows(ds, win) < usable_rows(ds, full)
    assert fit_final(ds, "market_linear", win)[2] == usable_rows(ds, win)
    a = walk_forward(ds, "market_linear", full)
    b = walk_forward(ds, "market_linear", TrainSettings(folds=4, min_move=0.25))
    assert a.index.equals(b.index) and not np.allclose(a.p_up, b.p_up)
    assert np.allclose(a.expected, b.expected)                           # return model still uses every row
    e = walk_forward(ds, "market_ensemble", full)
    assert e.index.equals(a.index) and e.p_up.between(0, 1).all()


def test_forecast_lists_unavailable_external_models(tmp_path):
    from src.forecast_engine import forecast, resolve_window
    ds, px, cal, ms = _ds_with_external(tmp_path, groups=("geopolitics",))
    req = resolve_window(cal, px.index[-1], "next", 5)
    out = forecast(px, cal, AstroSettings(), req, {}, 10,
                   ["market_linear", "market_geo_linear", "market_weather_linear"],
                   "market_linear", TrainSettings(folds=4, min_train_sessions=300),
                   {"geopolitics": ds.X["geopolitics"]}, ["weather is India-only"])
    rows = {r["model"]: r for r in out["models"]}
    assert "prob_up" in rows["market_geo_linear"]
    assert rows["market_weather_linear"]["status"].startswith("unavailable")
    assert "weather is India-only" in out["notes"]


def test_forecast_groups_by_exchange(monkeypatch):
    import src.risk_external as rx
    idx = pd.bdate_range("2020-01-01", periods=50)
    calls = []

    def fake(sessions, end, synthetic=False, seed=42, live=False, weather=True):
        calls.append((live, weather))
        use = rx.GEO_GLOBAL + rx.GEO_INDIA + (rx.WEATHER_COLS if weather else [])
        return pd.DataFrame(0.0, index=sessions, columns=use), {}
    monkeypatch.setattr(rx, "round2_features", fake)
    g, notes = rx.forecast_groups(idx, "NSE")
    assert set(g) == {"geopolitics", "weather"} and list(g["geopolitics"].columns) == rx.GEO_GLOBAL + rx.GEO_INDIA
    g, notes = rx.forecast_groups(idx, "NYSE")
    assert set(g) == {"geopolitics"} and list(g["geopolitics"].columns) == rx.GEO_GLOBAL and notes
    assert all(live for live, _ in calls)                                # dashboard uses the live, dated copies

    def broken(*a, **k):
        raise ConnectionError("offline")
    monkeypatch.setattr(rx, "round2_features", broken)
    g, notes = rx.forecast_groups(idx, "NSE")
    assert g == {} and "unavailable" in notes[0]                         # dashboard degrades, does not crash
