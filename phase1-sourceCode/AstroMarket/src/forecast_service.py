"""Orchestration used by the dashboard and CLI: data -> engine -> database records."""
import logging

import numpy as np
import pandas as pd

from . import repository as repo
from .ephemeris import AstroSettings
from .risk_external import forecast_groups
from .forecast_engine import build_dataset, compare, forecast, permutation_test, resolve_window, \
    sky_for_sessions, astro_adds_value
from .forecast_models import SPEC_BY_NAME
from .market_service import MarketService
from .training import TrainSettings

log = logging.getLogger(__name__)


def total_cost_bps(cfg) -> float:
    return float(cfg["backtest"]["cost_bps"]) + float((cfg.get("forecast") or {}).get("slippage_bps", 0))


def realized_return(px: pd.DataFrame, window: pd.DatetimeIndex) -> float | None:
    """Actual log return over the window, or None if it has not completed yet."""
    if window[-1] > px.index[-1] or window[0] not in px.index or window[-1] not in px.index:
        return None
    i0 = px.index.get_loc(window[0])
    if i0 == 0:
        return None
    return float(np.log(px["price"].iloc[px.index.get_loc(window[-1])] / px["price"].iloc[i0 - 1]))


def _register(con, spec, train: TrainSettings) -> int:
    """Stored model version = model code/parameters + training recipe, so results stay distinguishable."""
    return repo.register_model(con, spec.name, f"{spec.version}-t{train.key[:8]}", spec.family,
                               {"algo": spec.algo, "groups": spec.groups, **spec.params, "training": vars(train)})


def external_groups(ms: MarketService, sessions: pd.DatetimeIndex, exchange_code: str):
    """Geopolitics / weather groups for real data; none for synthetic prices (offline, and the dates
    of generated prices have no relation to real-world events)."""
    if getattr(ms.provider, "name", "") == "synthetic":
        return {}, ["Geopolitics/weather models are not run on synthetic prices."]
    return forecast_groups(sessions, exchange_code)


def category_ctx(ms: MarketService, symbol: str, exchange_code: str) -> dict:
    """Catalog-category groups; downloads are skipped for synthetic prices."""
    return {"exchange": exchange_code, "symbol": symbol, "fetch": getattr(ms.provider, "name", "") != "synthetic"}


def run_forecast(ms: MarketService, symbol: str, s: AstroSettings, as_of=None, mode="next", horizon=1,
                 date=None, start=None, end=None, models=None, primary="astro_stat", save=True,
                 train: TrainSettings | None = None) -> dict:
    cfg = ms.cfg
    fc = cfg.get("forecast") or {}
    px_full = ms.daily(symbol)
    as_of = px_full.index[-1] if as_of is None else pd.Timestamp(as_of)
    px_pit = px_full.loc[:as_of]                      # <- the only prices the engine ever sees
    if px_pit.empty:
        raise ValueError(f"No data on or before {as_of.date()}")
    # Exchange calendars are published in advance, so actual session dates may be used;
    # prices after the as-of session never reach the engine.
    cal = ms.calendar(symbol, px_full.index)
    req = resolve_window(cal, px_pit.index[-1], mode, horizon, date, start, end)
    # geopolitics / weather, each value re-dated to its publication date, so only what was public by as-of
    external, ext_notes = external_groups(ms, px_pit.index, cal.ex.code)
    ctx = category_ctx(ms, symbol, cal.ex.code)
    train = train or TrainSettings.from_cfg(cfg)
    out = forecast(px_pit, cal, s, req, fc, total_cost_bps(cfg), models or list(SPEC_BY_NAME), primary,
                   train, external, ext_notes, ctx)
    out["symbol"] = symbol
    out["realized"] = realized_return(px_full, req["window"])
    if save and out["status"] == "ok":
        inst = ms.instrument(symbol)
        for r in out["models"]:
            if "prob_up" not in r:
                continue
            mid = _register(ms.con, SPEC_BY_NAME[r["model"]], train)
            conf = r["confidence"]
            repo.save_prediction(ms.con, {
                "instrument_id": inst["id"], "model_version_id": mid,
                "as_of_session": str(out["as_of"].date()), "window_start": str(req["window"][0].date()),
                "window_end": str(req["window"][-1].date()), "gap_sessions": out["gap"],
                "horizon_sessions": out["horizon"], "zodiac": s.zodiac, "ayanamsa": s.ayanamsa_label,
                "direction": r["direction"], "prob_up": r["prob_up"], "expected_return": r["expected_return"],
                "interval_low": r["interval_low"], "interval_high": r["interval_high"],
                "expected_vol": out["expected_vol"], "risk_category": out["risk"],
                "confidence": conf.get("hit_rate") if conf.get("calibrated") else None,
                "evidence": r["evidence"], "generated_at": repo.utc_iso(out["generated_at"]),
                "realized_return": out["realized"],
                "details_json": {"metrics": {k: v for k, v in r["metrics"].items() if k != "backtest"},
                                 "notes": out["notes"], "n_train": out["n_train"]}})
    return out


def history_dataset(ms: MarketService, symbol: str, s: AstroSettings, gap: int, horizon: int, start=None):
    """Every group (market, astro, external, categories) over the full history (or from `start`)."""
    px = ms.daily(symbol)
    if start is not None:
        px = px.loc[pd.Timestamp(start):]
    cal = ms.calendar(symbol, px.index)
    future = cal.next_sessions(px.index[-1], gap + 1)
    sky_all = sky_for_sessions(px.index.append(future), cal, s, (ms.cfg.get("forecast") or {}).get("session_anchor", "open"))
    external, ext_notes = external_groups(ms, px.index, cal.ex.code)
    return build_dataset(px, sky_all, gap, horizon, external, ext_notes, category_ctx(ms, symbol, cal.ex.code))


def run_backtest(ms: MarketService, symbol: str, s: AstroSettings, gap: int, horizon: int, models: list[str],
                 permutations: int = 0, start=None, save=True, train: TrainSettings | None = None) -> dict:
    """Comparative walk-forward backtest over the full history (or from `start`)."""
    cfg = ms.cfg
    fc = cfg.get("forecast") or {}
    ds = history_dataset(ms, symbol, s, gap, horizon, start)
    px = ds.px
    train, cost = train or TrainSettings.from_cfg(cfg), total_cost_bps(cfg)
    res = compare(ds, models, train, cost)
    from .catalog_features import coverage_table
    out = {"results": res, "astro_vs": astro_adds_value(res, horizon), "dataset": ds, "training": train.describe(),
           "skipped": [m for m in models if m not in res], "notes": ds.notes,
           "coverage": coverage_table({k: ds.X[k] for k in ds.category_keys()})}
    if permutations and "astro_stat" in models:
        out["permutation"] = permutation_test(ds, "astro_stat", train, permutations)
    if save:
        inst = ms.instrument(symbol)
        for name, r in res.items():
            spec = r["spec"]
            mid = _register(ms.con, spec, train)
            key = repo.stable_hash([symbol, spec.version, gap, horizon, s.zodiac, s.ayanamsa_label, s.node,
                                    s.orb, str(px.index[0].date()), str(px.index[-1].date()), train.key, cost,
                                    fc.get("session_anchor", "open")])
            m = r["metrics"]
            repo.save_backtest(ms.con, key, inst["id"], mid, gap, horizon, s.zodiac, s.ayanamsa_label,
                               px.index[0], px.index[-1], m["eval_start"], m["eval_end"], train.folds, cost, m)
    return out


def persist_sky(ms: MarketService, timestamps, s: AstroSettings, events: pd.DataFrame | None = None) -> None:
    """Store raw ephemeris positions (and optionally derived events) for reproducibility."""
    from .ephemeris import positions
    repo.source_id(ms.con, f"swisseph-{s.ephemeris}", "ephemeris", False,
                   "Swiss Ephemeris via pyswisseph (Moshier built-in unless .se1 files configured)")
    repo.save_positions(ms.con, positions(timestamps, s), s.zodiac, s.ayanamsa_label, s.ephemeris)
    if events is not None and len(events):
        repo.save_events(ms.con, events, s.zodiac, s.ayanamsa_label)


def refresh_realized(ms: MarketService, symbol: str) -> int:
    """Fill realized_return for stored predictions whose window has since closed."""
    preds = repo.list_predictions(ms.con, symbol)
    pending = preds[preds.realized_return.isna()]
    if pending.empty:
        return 0
    px = ms.daily(symbol)
    cal = ms.calendar(symbol, px.index)
    n = 0
    for r in pending.itertuples():
        window = cal.sessions_between(r.window_start, r.window_end)
        if len(window) == 0:
            continue
        val = realized_return(px, window)
        if val is not None:
            repo.set_realized(ms.con, r.id, val)
            n += 1
    return n


def run_vol_backtest(ms: MarketService, symbol: str, s: AstroSettings, gap: int, horizon: int,
                     groups: list[str] | None = None, start=None) -> dict:
    """Volatility target: HAR benchmark vs HAR + each data category, identical walk-forward dates."""
    from .vol_engine import candidate_groups, vol_compare
    ds = history_dataset(ms, symbol, s, gap, horizon, start)
    available = candidate_groups(ds)
    chosen = [g for g in (groups or available) if g in available]
    out = vol_compare(ds, chosen, TrainSettings.from_cfg(ms.cfg).folds)
    out.update({"notes": ds.notes, "groups_tested": chosen,
                "groups_missing": [g for g in (groups or []) if g not in available]})
    return out


def run_vol_forecast(ms: MarketService, symbol: str, s: AstroSettings, gap: int, horizon: int,
                     group: str = "c05") -> dict | None:
    """Live expected volatility over the next `horizon` sessions from the validated HAR+`group`
    model (see src/vol_engine.py:live_forecast) - `None` when the group isn't available for this
    instrument or there's too little history to fit reliably."""
    from .vol_engine import live_forecast
    ds = history_dataset(ms, symbol, s, gap, horizon)
    return live_forecast(ds, group)


def completed_sessions(ms: MarketService, symbol: str, px: pd.DataFrame) -> pd.DatetimeIndex:
    """Session dates with a final close: today's bar is dropped while the exchange is still open."""
    from .exchanges import is_open_now
    ex = ms.calendar(symbol, px.index).ex
    today = pd.Timestamp.now(tz=ex.tz).normalize().tz_localize(None)
    idx = px.index
    return idx[idx < today] if (len(idx) and idx[-1] >= today and is_open_now(ex)) else idx


def period_sessions(ms: MarketService, symbol: str, preset_sessions: int | None = None,
                    start=None, end=None) -> pd.DatetimeIndex:
    """The last `preset_sessions` completed sessions, or the completed sessions in [start, end]."""
    px = ms.daily(symbol)
    done = completed_sessions(ms, symbol, px)
    if preset_sessions:
        return done[-int(preset_sessions):]
    return done[(done >= pd.Timestamp(start)) & (done <= pd.Timestamp(end))]


def run_hindcast(ms: MarketService, symbol: str, s: AstroSettings, window: pd.DatetimeIndex, model: str,
                 mode: str = "strict") -> dict:
    """Actual prices for `window` next to predictions made without those prices (see src/hindcast.py)."""
    from .hindcast import hindcast
    cfg = ms.cfg
    px = ms.daily(symbol)
    cal = ms.calendar(symbol, px.index)
    return hindcast(px, cal, s, window, model, mode, (cfg.get("forecast") or {}).get("session_anchor", "open"),
                    train=TrainSettings.from_cfg(cfg),
                    external_fn=lambda sessions: external_groups(ms, sessions, cal.ex.code),
                    category_ctx=category_ctx(ms, symbol, cal.ex.code))
