"""Streamlit UI: Market Prediction dashboard screens."""
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from . import repository as repo
from .analysis import condition_stats, conditions, event_study, forward_returns
from .astrology import chart, detect_events, readable, sky_table
from .ephemeris import AstroSettings
from .exchanges import EXCHANGES, session_time_utc
from .forecast_models import (ALGORITHMS, COMPARISON_MODELS, DEFAULT_MODELS, SELECTABLE_GROUPS, SPEC_BY_NAME,
                              custom_model_name)
from .forecast_engine import common_dates
from .forecast_service import (period_sessions, persist_sky, refresh_realized, run_backtest, run_forecast,
                               run_hindcast, run_vol_backtest, total_cost_bps)
from .market_service import MarketService, resample_bars
from .market_providers import ProviderError
from .training import TrainSettings
from . import ui_components as ui

try:  # optional: live type-ahead dropdown for symbol search
    from streamlit_searchbox import st_searchbox
except ImportError:
    st_searchbox = None

LOGO = Path(__file__).resolve().parent.parent / "assets" / "logo.svg"
PRESETS = {"^NSEI": "Nifty 50", "^BSESN": "Sensex", "^NSEBANK": "Nifty Bank", "RELIANCE.NS": "Reliance",
           "TCS.NS": "TCS", "HDFCBANK.NS": "HDFC Bank", "INFY.NS": "Infosys", "^GSPC": "S&P 500", "AAPL": "Apple"}
BADGE = {"real-time": ("green", "Real-time"), "delayed": ("orange", "Delayed"),
         "end-of-day": ("blue", "End-of-day"), "synthetic": ("violet", "Synthetic data")}
EVIDENCE = {"validated": ("green", "Validated edge (out-of-sample)"),
            "experimental": ("orange", "Experimental: no validated edge"),
            "insufficient": ("red", "Insufficient evidence"), "baseline": ("gray", "Baseline")}
SUGGESTIONS = 5
DISCLAIMER = ("Research tool. No data category (market, macro, geopolitical, weather or astrological) is a "
              "guaranteed predictor of prices; any relationship shown is a historical correlation, not a cause. "
              "Nothing here is investment advice, no profit is implied and no trades are executed.")


def pct(log_r):
    return np.expm1(log_r) if log_r is not None else None


def fmt_pct(log_r, signed=True):
    if log_r is None or (isinstance(log_r, float) and np.isnan(log_r)):
        return "-"
    return f"{pct(log_r):+.2%}" if signed else f"{pct(log_r):.2%}"


# ---------------------------------------------------------------- cached loaders
def data_version(ms, px) -> str:
    """Cache key part for model results: last price date + the shared training recipe, so changing
    either (new data, or forecast settings in config.yaml) re-runs every page's models."""
    return f"{px.index[-1].date()}|{TrainSettings.from_cfg(ms.cfg).key}"


@st.cache_resource
def _service(_cfg, key):
    return MarketService(_cfg)


@st.cache_data(ttl=900, show_spinner="Loading daily prices...")
def _daily(_ms, symbol, exchange):
    _ms.instrument(symbol, exchange)
    return _ms.daily(symbol)


@st.cache_data(ttl=120, show_spinner="Loading intraday prices...")
def _intraday(_ms, symbol, interval, days):
    return _ms.intraday(symbol, interval, days)


@st.cache_data(ttl=3600, show_spinner=False)
def _provider_search(_ms, provider, query):
    return _ms.provider.search(query)


def suggest_symbols(ms, known: dict, term: str, limit: int = SUGGESTIONS) -> list[tuple[str, str]]:
    """Type-ahead: local symbols (presets + saved) first, then provider search; (label, symbol) pairs."""
    t = term.strip().upper()
    if not t:
        return []
    local = [(sym, name) for sym, name in known.items() if t in sym.upper() or t in str(name).upper()]
    local.sort(key=lambda r: not (r[0].upper().startswith(t) or str(r[1]).upper().startswith(t)))
    remote = []
    if len(t) >= 2:
        hits = _provider_search(ms, ms.provider.name, term.strip())
        remote = [(h["symbol"], f"{h['name']} ({h['exchange']})" if h.get("exchange") else h["name"])
                  for h in hits if h.get("name")]
    out, seen = [], set()
    for sym, name in local + remote:
        if sym not in seen:
            seen.add(sym)
            out.append((f"{sym} - {name}", sym))
            st.session_state.setdefault("symbol_names", {})[sym] = name
    return out[:limit]


@st.cache_data(ttl=60, show_spinner=False)
def _latest(_ms, symbol):
    return _ms.latest(symbol)


@st.cache_data(show_spinner="Computing planetary positions...")
def _sky_sessions(_ms, symbol, _sessions, key, ex_code, s: AstroSettings, anchor):
    ts = session_time_utc(EXCHANGES[ex_code], _sessions, anchor)
    sky = sky_table(ts, s, index=_sessions)
    try:
        persist_sky(_ms, ts, s)
    except Exception as e:  # persistence is best-effort; analysis still works
        st.warning(f"Could not store positions: {e}")
    return sky


@st.cache_data(show_spinner="Computing ephemeris timeline...")
def _sky_daily(start, end, s: AstroSettings):
    ts = pd.date_range(start, end, freq="D", tz="UTC")
    return sky_table(ts, s, index=ts.tz_localize(None))


def _train(ms, tune: bool) -> TrainSettings | None:
    """None keeps the default (untuned) recipe from config.yaml; a tuned copy only when asked."""
    from dataclasses import replace
    return replace(TrainSettings.from_cfg(ms.cfg), tune=True) if tune else None


@st.cache_data(show_spinner="Running walk-forward validation and forecast (10-60 s, longer if tuning is on)...")
def _forecast(_ms, symbol, s, as_of, mode, horizon, date, start, end, models, primary, data_end, tune=False):
    return run_forecast(_ms, symbol, s, as_of, mode, horizon, date, start, end, list(models), primary,
                        train=_train(_ms, tune))


@st.cache_data(show_spinner="Backtesting all models walk-forward (longer if tuning is on)...")
def _backtest(_ms, symbol, s, gap, horizon, models, perms, start, data_end, tune=False):
    out = run_backtest(_ms, symbol, s, gap, horizon, list(models), perms, start, train=_train(_ms, tune))
    out.pop("dataset")
    return out


@st.cache_data(show_spinner="Predicting the period without its prices (10-90 s)...")
def _hindcast(_ms, symbol, s, window, model, mode, data_end):
    return run_hindcast(_ms, symbol, s, pd.DatetimeIndex(window), model, mode)


@st.cache_data(show_spinner="Backtesting volatility models on identical dates (20-60 s)...")
def _vol_backtest(_ms, symbol, s, gap, horizon, groups, data_end):
    return run_vol_backtest(_ms, symbol, s, gap, horizon, list(groups))


# ---------------------------------------------------------------- charts
CANDLE_COLORS = ("#2e9e5b", "#d1495b")            # actual: up, down
PREDICTED_COLORS = ("#8fd4ad", "#f0a3ad")         # predicted: lighter shades of the same green / red


def price_chart(df: pd.DataFrame, height=340, shade: pd.DataFrame | None = None, domain=None,
                colors=CANDLE_COLORS, change_labels=False):
    """change_labels: write "+12.30 (+0.45%)" above each candle (needs `change` and `pct_change` columns)."""
    d = df.drop(columns="ts_utc", errors="ignore").rename_axis("t").reset_index()
    tips = [alt.Tooltip("t:T", title="Time"), alt.Tooltip("open:Q", format=",.2f"),
            alt.Tooltip("high:Q", format=",.2f"), alt.Tooltip("low:Q", format=",.2f"),
            alt.Tooltip("close:Q", format=",.2f")]
    if "volume" in d:
        tips.append(alt.Tooltip("volume:Q", format=",.0f"))
    if "change" in d:
        tips.append(alt.Tooltip("change:Q", format="+,.2f", title="Change"))
    if "pct_change" in d:
        tips.append(alt.Tooltip("pct_change:Q", format="+.2%", title="Change %"))
    base = alt.Chart(d).encode(x=alt.X("t:T", title=None), tooltip=tips)
    scale = alt.Scale(zero=False, domain=domain) if domain else alt.Scale(zero=False)
    y = alt.Y("low:Q", scale=scale, title="Price")
    if len(d) <= 300:
        color = alt.condition("datum.open <= datum.close", alt.value(colors[0]), alt.value(colors[1]))
        ch = base.mark_rule().encode(y=y, y2="high:Q", color=color) + \
            base.mark_bar(size=max(1, int(600 / max(len(d), 1)))).encode(y="open:Q", y2="close:Q", color=color)
        if change_labels and {"change", "pct_change"} <= set(d) and len(d) <= 30:
            d["change_label"] = [f"{c:+,.2f} ({p:+.2%})" if pd.notna(c) else "" for c, p in zip(d["change"], d["pct_change"])]
            up = alt.condition("datum.change >= 0", alt.value(CANDLE_COLORS[0]), alt.value(CANDLE_COLORS[1]))
            rotate = len(d) > 6                          # vertical labels once horizontal ones would overlap
            ch += alt.Chart(d).mark_text(**({"angle": 270, "align": "left", "baseline": "middle", "dx": 5} if rotate
                                            else {"align": "center", "baseline": "bottom", "dy": -4}),
                                         fontSize=10).encode(
                x="t:T", y=alt.Y("high:Q", scale=scale), text="change_label:N", color=up, tooltip=tips)
    else:
        ch = base.mark_line(strokeWidth=1.2).encode(y=alt.Y("close:Q", scale=alt.Scale(zero=False), title="Price"))
    if shade is not None and len(shade):
        rect = alt.Chart(shade).mark_rect(opacity=0.18, color="#8a63d2").encode(x="start:T", x2="end:T")
        ch = rect + ch
    return ch.properties(height=height).interactive(bind_y=False)


def spans(mask: pd.Series) -> pd.DataFrame:
    """Contiguous True periods of a boolean series -> start/end rows (for shading)."""
    m = mask.astype(int).to_numpy()
    if not m.any():
        return pd.DataFrame(columns=["start", "end"])
    edges = np.diff(np.r_[0, m, 0])
    s, e = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1
    return pd.DataFrame({"start": mask.index[s], "end": mask.index[e] + pd.Timedelta(days=1)})


def badge(kind: str, mapping: dict):
    color, text = mapping.get(kind, ("gray", kind))
    st.badge(text, color=color)


# ---------------------------------------------------------------- help texts (from the help_texts table)
_HELP: dict[str, str] = {}


def tip(key) -> str | None:
    """Tooltip for a control id or table column name; None when the table has no entry."""
    k = str(key).strip().lower()
    for cand in (k, k.rsplit(".", 1)[-1], re.sub(r"^(strategy|buy_hold)_", "", k),
                 "ts_utc" if "ts_utc" in k else "", re.split(r"[ _]", k)[-1]):
        if cand in _HELP:
            return _HELP[cand]
    return None


def table(where, df, **kw):
    """st.dataframe with a tooltip on every column header that has a help text."""
    cfg = {c: st.column_config.Column(help=h) for c in map(str, df.columns) if (h := tip(c))}
    return where.dataframe(df, column_config={**cfg, **kw.pop("column_config", {})}, **kw)


def heading(text, key, level=4):
    st.markdown(f"{'#' * level} {text}", help=tip(key))


def load_prices(ms, symbol, ex):
    try:
        return _daily(ms, symbol, ex.code)
    except ProviderError as e:
        st.error(f"Market data unavailable for {symbol}: {e}")
        st.stop()


# ---------------------------------------------------------------- screens
def screen_overview(ms, symbol, ex, tz, s, px):
    inst = ms.instrument(symbol)
    st.markdown(f"#### {inst.get('name') or symbol} · `{symbol}` · {ex.code}"
                + (f" · {inst['sector']}" if inst.get("sector") else ""))
    try:
        q = _latest(ms, symbol)
        c1, c2, c3 = st.columns([1, 1, 2])
        c1.metric("Latest price", f"{q['price']:,.2f} {q['currency'] or ''}", f"{q['change_pct']:+.2%}",
                  help=tip("latest_price"))
        with c2:
            badge(q["label"], BADGE)
            st.caption("Market open" if q["market_open"] else "Market closed")
        c3.caption(f"Last bar: {q['ts_exchange']:%Y-%m-%d %H:%M %Z} ({q['ts_utc'].tz_convert(tz):%H:%M} {tz}) · "
                   f"age {str(q['age']).split('.')[0]} · source **{q['source']}**. {q['source_notes']}")
    except ProviderError as e:
        st.warning(f"Latest quote unavailable: {e}")

    period = st.segmented_control("Period", ["1D", "5D", "1M", "6M", "1Y", "5Y", "Max"], default="6M", help=tip("period"))
    bars = st.segmented_control("Bars", ["Daily", "Weekly", "Monthly", "Yearly"], default="Daily",
                                disabled=period in ("1D", "5D"), help=tip("bars"))
    if period in ("1D", "5D"):
        try:
            intr = _intraday(ms, symbol, "5m" if period == "1D" else "15m", 5 if period == "1D" else 9)
            days = intr.index.normalize().unique()[-(1 if period == "1D" else 5):]
            d = intr[intr.index.normalize().isin(days)].copy()
            d.index = d["ts_utc"].dt.tz_convert(tz).dt.tz_localize(None)
            st.altair_chart(price_chart(d), width="stretch")
            st.caption(f"Intraday bars, times in {tz}. Freshness: delayed (provider has no real-time guarantee).")
        except ProviderError as e:
            st.info(f"Intraday data unavailable: {e}")
    else:
        start = {"1M": 31, "6M": 183, "1Y": 366, "5Y": 1827}.get(period)
        d = px if start is None else px[px.index >= px.index[-1] - pd.Timedelta(days=start)]
        rule = {"Weekly": "W-FRI", "Monthly": "ME", "Yearly": "YE"}.get(bars)
        if rule:
            d = resample_bars(d, rule)
        st.altair_chart(price_chart(d), width="stretch")
        st.caption("End-of-day bars (completed sessions only). Weekly/monthly/yearly bars aggregate actual sessions.")

    heading("Sky now", "sky_now")
    now = pd.Timestamp.now(tz="UTC")
    sky_block(now, s, tz)
    heading("Upcoming astrological events (next 30 days)", "upcoming_events")
    up = _sky_daily(now.normalize().tz_localize(None), now.normalize().tz_localize(None) + pd.Timedelta(days=31), s)
    ev = detect_events(up, s.zodiac)
    table(st, ev.assign(date=ev.date.dt.date), hide_index=True, width="stretch", height=260)


def sky_block(ts, s, tz):
    c = chart(ts, s)
    lun = c["lunar"]
    a, b, d, e = st.columns(4)
    a.metric("Moon phase", lun["phase"], f"{lun['illumination']:.0%} illuminated", delta_color="off",
             help=tip("moon_phase"))
    b.metric("Tithi", f"{lun['tithi']} {lun['tithi_name']}", f"{lun['paksha']} paksha", delta_color="off",
             help=f"{tip('tithi') or ''}\n\n{tip('paksha') or ''}".strip())
    d.metric("Moon sign", lun["moon_sign"], help=tip("moon_sign"))
    e.metric("Moon nakshatra", lun["moon_nakshatra"], help=tip("moon_nakshatra"))
    st.caption(f"Computed for {pd.Timestamp(ts).tz_convert(tz):%Y-%m-%d %H:%M %Z} · {s.zodiac}"
               + (f" ({s.ayanamsa} ayanamsa)" if s.zodiac == "sidereal" else "") + f" · {s.ephemeris} ephemeris")
    p = c["positions"]
    l, r = st.columns(2)
    l.markdown("**Astronomical** (measured quantities)", help=tip("astronomical_positions"))
    table(l, p[["body", "longitude", "latitude", "distance_au", "speed_deg_day", "declination"]].round(3),
                hide_index=True, width="stretch")
    r.markdown("**Astrological interpretation** (conventions)", help=tip("astrological_interpretation"))
    table(r, p[["body", "sign", "degree_in_sign", "nakshatra", "pada", "retrograde"]].round(2),
                hide_index=True, width="stretch")
    with st.expander(f"Aspects within {s.orb}° orb ({len(c['aspects'])})"):
        table(st, c["aspects"], hide_index=True, width="stretch")


def screen_history(ms, symbol, ex, tz, s, px, anchor):
    sky = _sky_sessions(ms, symbol, px.index, (str(px.index[0]), str(px.index[-1]), len(px)), ex.code, s, anchor)
    lo, hi = px.index[0].date(), px.index[-1].date()
    c1, c2, c3 = st.columns([2, 1, 2])
    rng = c1.date_input("Date range", (max(lo, hi - pd.Timedelta(days=365)), hi), min_value=lo, max_value=hi,
                        key="hist_range", help=tip("date_range"))
    horizon = c2.selectbox("Forward return (sessions)", [1, 3, 5, 10, 20], key="hist_h",
                           help=tip("forward_return_sessions"))
    cond = conditions(sky, s.zodiac)
    hl = c3.selectbox("Highlight condition", ["(none)"] + [c for c in cond.columns if set(cond[c]) <= {"yes", "no"}],
                     help=tip("highlight_condition"))
    a, b = (pd.Timestamp(rng[0]), pd.Timestamp(rng[1])) if len(rng) == 2 else (px.index[0], px.index[-1])
    sel = px.loc[a:b]
    shade = spans(cond.loc[a:b, hl] == "yes") if hl != "(none)" else None
    st.altair_chart(price_chart(sel, shade=shade), width="stretch")
    if hl != "(none)":
        st.caption(f"Shaded: sessions where **{hl}** (sky at session {anchor}, {ex.tz}).")

    with st.expander("Session-level data: prices + sky at session " + anchor, expanded=False):
        t = pd.concat([sel[["ts_utc", "open", "high", "low", "close", "adj_close", "volume", "pct_change"]]
                       .rename(columns={"ts_utc": "close_ts_utc"}),
                       sky.loc[a:b, ["ts_utc"]].rename(columns={"ts_utc": f"sky_ts_utc ({anchor})"}),
                       readable(sky.loc[a:b], s.zodiac)], axis=1)
        table(st, t.sort_index(ascending=False), width="stretch", height=380)

    heading(f"Observed correlations: {horizon}-session forward return by condition (full history)",
            "observed_correlations")
    fwd = forward_returns(px["log_ret"], 0, horizon).shift(1)   # return of sessions j..j+h-1
    stats_df = condition_stats(fwd, cond, float(ms.cfg["validation"]["fdr_alpha"]))
    if stats_df.empty:
        st.info("Not enough sessions for condition tests.")
    else:
        n_sig = int(stats_df.significant_fdr.sum())
        naive = int((stats_df.p_value < 0.05).sum())
        st.write(f"{len(stats_df)} condition tests: **{naive}** look significant at p<0.05 before correction "
                 f"(about {0.05 * len(stats_df):.0f} expected by pure chance); **{n_sig}** survive "
                 f"Benjamini-Hochberg FDR. Overlapping multi-session windows make p-values optimistic.")
        table(st, stats_df.round(4), hide_index=True, width="stretch", height=320)

    heading("Returns around astrological events (event study)", "event_study")
    days = _sky_daily(px.index[0] - pd.Timedelta(days=1), px.index[-1], s)
    ev = detect_events(days, s.zodiac)
    ev["key"] = ev.event_type + np.where(ev.body1.isin(["moon"]) & ev.event_type.isin(["new_moon", "full_moon"]),
                                         "", " · " + ev.body1.fillna("") + np.where(ev.body2.notna() &
                                         ~ev.event_type.isin(["new_moon", "full_moon"]), "-" + ev.body2.fillna(""), ""))
    counts = ev.key.value_counts()
    choices = counts[counts >= 8].index.tolist()
    k = st.selectbox("Event", choices, index=choices.index("new_moon") if "new_moon" in choices else 0,
                     help=tip("event"))
    es = event_study(px["log_ret"], ev.loc[ev.key == k, "date"], 5, 10)
    if es["n"] < 8:
        st.info("Too few events in the price history for this event type.")
    else:
        path = es["path"].melt("session", ["mean_cum_ret", "unconditional"], "series", "cum")
        band = alt.Chart(es["path"]).mark_area(opacity=.2).encode(x="session:Q", y="ci_low:Q", y2="ci_high:Q")
        line = alt.Chart(path).mark_line(point=True).encode(
            x=alt.X("session:Q", title="Sessions relative to event (0 = first session on/after)"),
            y=alt.Y("cum:Q", axis=alt.Axis(format="%"), title="Mean cumulative log return"),
            color="series:N", tooltip=["session", alt.Tooltip("cum:Q", format="+.3%"), "series"])
        st.altair_chart((band + line).properties(height=280), width="stretch")
        st.caption(f"{es['n']} events. Excess return at +10 vs unconditional drift: {es['end_excess']:+.3%} "
                   f"(t={es['end_t']:.2f}, p={es['end_p']:.3f}; single test, not corrected for the "
                   f"{len(choices)} event types available). Shaded band: 95% CI.")


def screen_forecast(ms, symbol, ex, tz, s, px):
    fc = ms.cfg.get("forecast") or {}
    c1, c2, c3 = st.columns(3)
    as_of = c1.date_input("As-of session (forecast timestamp)", px.index[-1].date(),
                          min_value=px.index[0].date(), max_value=px.index[-1].date(),
                          help=tip("as_of"))
    mode = c2.radio("Target", ["Next N sessions", "Custom date", "Custom date range"], help=tip("target"))
    horizon, date, start, end = 1, None, None, None
    if mode == "Next N sessions":
        horizon = c3.selectbox("Sessions", fc.get("horizons", [1, 3, 5, 10, 20]), help=tip("target_sessions"))
    elif mode == "Custom date":
        date = c3.date_input("Target date", pd.Timestamp(as_of) + pd.Timedelta(days=7),
                            help=tip("target_date"))
    else:
        r = c3.date_input("Target range", (pd.Timestamp(as_of) + pd.Timedelta(days=1),
                                           pd.Timestamp(as_of) + pd.Timedelta(days=14)), help=tip("target_range"))
        if len(r) == 2:
            start, end = r
    m1, m2 = st.columns([1, 2])
    primary = m1.selectbox("Headline model", ["astro_stat", "astro_lgbm", "hybrid_lgbm", "market_linear"],
                           format_func=lambda n: SPEC_BY_NAME[n].label, help=tip("headline_model"))
    models = m2.multiselect("Compare models", list(SPEC_BY_NAME), DEFAULT_MODELS,
                            format_func=lambda n: SPEC_BY_NAME[n].label, help=tip("models"))
    if primary not in models:
        models = [primary] + models
    tune = st.checkbox("Tune hyperparameters (slower)", key="fc_tune", help=tip("tune_hyperparams"))
    if not st.button("Generate forecast", type="primary"):
        st.info("Choose the target and press **Generate forecast**. Each model is re-validated walk-forward "
                "on data before the as-of session, so this takes a little while.")
        return
    if mode == "Custom date range" and start is None:
        st.warning("Select both ends of the range."); return
    kind = {"Next N sessions": "next", "Custom date": "date", "Custom date range": "range"}[mode]
    try:
        o = _forecast(ms, symbol, s, pd.Timestamp(as_of), kind, horizon, date, start, end, tuple(models), primary,
                      data_version(ms, px), tune)
    except ValueError as e:
        st.error(str(e)); return
    render_forecast(o, symbol, tz, s, px)


def render_forecast(o, symbol, tz, s, px):
    w = o["window"]
    st.markdown(f"### {symbol}: {w[0]:%a %d %b %Y}" + (f" to {w[-1]:%a %d %b %Y}" if len(w) > 1 else ""))
    st.caption(f"Generated {o['generated_at'].tz_convert(tz):%Y-%m-%d %H:%M %Z} · as-of close "
               f"{o['as_of']:%Y-%m-%d} · {o['horizon']} trading session(s)"
               + (f" after a gap of {o['gap']}" if o["gap"] else "") + f" · {o['n_train']} training sessions")
    if o.get("training"):
        st.caption(o["training"] + " (the same recipe on every page)")
    for n in o["notes"]:
        st.info(n)
    if o["status"] != "ok":
        st.error(f"Forecast unavailable. {o['message']}")
        return
    p = o["primary"]
    badge(p["evidence"], EVIDENCE)
    if tip("evidence_badge"):
        st.caption(tip("evidence_badge"))
    a, b, c, d = st.columns(4)
    arrow = {"bullish": "▲ Bullish", "bearish": "▼ Bearish", "neutral": "● Neutral"}[p["direction"]]
    a.metric(f"Direction ({p['label']})", arrow, f"P(up) = {p['prob_up']:.1%}", delta_color="off",
             help=tip("direction"))
    b.metric("Expected return", fmt_pct(p["expected_return"]),
             f"80% interval {fmt_pct(p['interval_low'])} to {fmt_pct(p['interval_high'])}", delta_color="off",
             help=tip("expected_return"))
    c.metric("Expected volatility", f"{o['expected_vol']:.2%}", f"risk: {o['risk']}", delta_color="off",
             help=tip("expected_volatility"))
    conf = p["confidence"]
    if conf.get("calibrated"):
        d.metric("Validated hit rate", f"{conf['hit_rate']:.1%}",
                 f"95% CI {conf['ci_low']:.0%}-{conf['ci_high']:.0%}, n={conf['n']}", delta_color="off",
                 help=tip("validated_hit_rate"))
    else:
        d.metric("Confidence", "Not calibrated", conf.get("note", ""), delta_color="off", help=tip("confidence"))
    m = p["metrics"]
    st.caption(f"Out-of-sample validation of this model at this horizon: accuracy {m['accuracy']:.1%} vs "
               f"{m['majority_rate']:.1%} by always predicting the majority direction; AUC {m['auc']:.3f}; "
               f"edge p-value {m['edge_p_value']:.3f} on {m['n_independent']} non-overlapping windows"
               + (f"; {m['confident_accuracy']:.1%} right on the {m['confident_share']:.0%} of days it made a "
                  "bullish/bearish (non-neutral) call. " if m.get("confident_share") else ". ")
               + "Interval = model estimate + 10th/90th percentile of its out-of-sample errors. "
               "Volatility = EWMA daily volatility × √sessions; risk category = its percentile in this history.")
    if p["evidence"] != "validated":
        st.warning("This forecast is **experimental**: the model did not show a statistically significant "
                   "out-of-sample edge over the naive baseline, so its direction should be read as noise.")
    if o.get("realized") is not None:
        hit = (o["realized"] > 0) == (p["prob_up"] > 0.5)
        st.success(f"Replay: actual return over the window was **{fmt_pct(o['realized'])}** "
                   f"({'direction correct' if hit else 'direction wrong'}). A single outcome says little "
                   f"about skill; see Backtesting Results.")

    price_section(o, p, px)

    heading("All models for this target", "all_models")
    rows = [{"model": r["label"], "family": r.get("family"), "direction": r.get("direction"),
             "P(up)": r.get("prob_up"), "expected": pct(r.get("expected_return")),
             "low 80%": pct(r.get("interval_low")), "high 80%": pct(r.get("interval_high")),
             "expected price": o["last_close"] * np.exp(r["expected_return"]),
             "price low 80%": o["last_close"] * np.exp(r["interval_low"]),
             "price high 80%": o["last_close"] * np.exp(r["interval_high"]),
             "evidence": r.get("evidence"), "val. accuracy": r.get("metrics", {}).get("accuracy"),
             "accuracy vs majority": r.get("metrics", {}).get("accuracy_edge"),
             "confident-call accuracy": r.get("metrics", {}).get("confident_accuracy"),
             "val. AUC": r.get("metrics", {}).get("auc"), "edge p": r.get("metrics", {}).get("edge_p_value"),
             "version": r.get("version")} for r in o["models"] if "prob_up" in r]
    table(st, pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
    if p.get("selected_features"):
        st.caption("Features kept by the L1 penalty (largest first): " + ", ".join(p["selected_features"]))
    elif p["model"] == "astro_stat":
        st.caption("The L1 penalty removed every astro feature: none carried enough signal to keep, so the "
                   "model falls back to the base rate.")

    heading("Astrological conditions during the target window", "window_conditions")
    ws = o["window_sky"]
    table(st, readable(ws, s.zodiac), width="stretch")
    days = sky_table(pd.date_range(o["as_of"], w[-1] + pd.Timedelta(days=1), tz="UTC"), s)
    days.index = days.index.tz_localize(None)
    ev = detect_events(days, s.zodiac)
    ev = ev[(ev.date > o["as_of"]) & (ev.date <= w[-1] + pd.Timedelta(days=1))]
    if len(ev):
        table(st, ev.assign(date=ev.date.dt.date)[["date", "detail"]], hide_index=True, width="stretch")

    heading("Historical analogues (similar fast-planet skies)", "analogues")
    an = o["analogues"]
    if an["status"] != "ok":
        st.warning(f"Analogue evidence unavailable: {an['message']}")
    else:
        a, b, c = st.columns(3)
        a.metric("Analogue sessions", an["n"], help=tip("analogue_sessions"))
        b.metric("Mean return (analogues)", fmt_pct(an["mean"]), f"all sessions {fmt_pct(an['base_mean'])}",
                 delta_color="off", help=tip("analogue_mean_return"))
        c.metric("Up share (analogues)", f"{an['hit_rate']:.0%}", f"all sessions {an['base_hit_rate']:.0%}",
                 delta_color="off", help=tip("analogue_up_share"))
        st.caption(f"Welch t-test vs all sessions: p = {an['p_value']:.3f}; {an['evidence']}. Analogues match "
                   "Sun, Moon, Mercury, Venus, Mars positions, lunar phase and retrogrades; matching the Sun "
                   "also matches the time of year, so seasonality is an alternative explanation.")
        t = an["table"].assign(forward_return=lambda d: np.expm1(d.forward_return))
        hist = alt.Chart(t).mark_bar(opacity=.7).encode(
            x=alt.X("forward_return:Q", bin=alt.Bin(maxbins=30), axis=alt.Axis(format="%"), title="Return"),
            y="count()")
        st.altair_chart(hist.properties(height=200), width="stretch")
        with st.expander("Analogue dates"):
            table(st, t.assign(date=t.date.dt.date).round(4), hide_index=True, width="stretch")

    with st.expander("Limitations and alternative explanations", expanded=True):
        st.markdown(
            "- Validation is walk-forward and out-of-sample, but past performance does not guarantee future "
            "behaviour; market regimes change.\n"
            "- Many astrological variables are tested; some will look predictive by chance (multiple testing).\n"
            "- Slow planets (Jupiter-Pluto, nodes) change over years and can act as hidden calendar/era proxies.\n"
            "- The Sun's sign repeats every year, so Sun-based patterns overlap with seasonality.\n"
            "- Future trading sessions are projected; unlisted exchange holidays shift session counts.\n"
            f"- {DISCLAIMER}")


def price_section(o, p, px):
    """Expected price level = as-of close x exp(expected log return), with the same 80% interval."""
    base, w, gap = o["last_close"], o["window"], o["gap"]
    exp_p, lo_p, hi_p = (base * np.exp(p[k]) for k in ("expected_return", "interval_low", "interval_high"))
    heading("Predicted price", "predicted_price")
    a, b, c = st.columns(3)
    a.metric(f"Close on {o['as_of']:%d %b %Y} (as-of)", f"{base:,.2f}", help=tip("as_of_close"))
    b.metric(f"Expected close on {w[-1]:%d %b %Y}", f"{exp_p:,.2f}", f"{exp_p - base:+,.2f}", delta_color="off",
             help=tip("expected_close"))
    c.metric("80% price range", f"{lo_p:,.2f} - {hi_p:,.2f}", help=tip("price_range_80"))

    hist = px.loc[:o["as_of"]].tail(60)
    rows = [{"date": o["as_of"], "series": "Forecast", "price": base, "low": base, "high": base}]
    for k, d in enumerate(w, 1):             # spread the window return evenly (in log terms) over its sessions
        f = k / len(w)
        rows.append({"date": d, "series": "Forecast", "price": base * np.exp(p["expected_return"] * f),
                     "low": base * np.exp(p["interval_low"] * f), "high": base * np.exp(p["interval_high"] * f)})
    proj = pd.DataFrame(rows)
    lines = [pd.DataFrame({"date": hist.index, "series": "Actual (history)", "price": hist["close"].to_numpy()})]
    after = px.loc[px.index.isin(w)]
    if len(after):                           # replay: show what actually happened
        lines.append(pd.DataFrame({"date": [o["as_of"], *after.index], "series": "Actual (outcome)",
                                   "price": [base, *after["close"].to_numpy()]}))
    lines = pd.concat(lines)
    tooltip = [alt.Tooltip("date:T"), "series:N", alt.Tooltip("price:Q", format=",.2f")]
    band = alt.Chart(proj).mark_area(opacity=.2, color="#8a63d2").encode(
        x="date:T", y=alt.Y("low:Q", scale=alt.Scale(zero=False), title="Price"), y2="high:Q",
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("low:Q", format=",.2f", title="low 80%"),
                 alt.Tooltip("high:Q", format=",.2f", title="high 80%")])
    fc = alt.Chart(proj).mark_line(strokeDash=[6, 4], point=True, color="#8a63d2").encode(
        x="date:T", y="price:Q", tooltip=tooltip)
    act = alt.Chart(lines).mark_line().encode(
        x=alt.X("date:T", title=None), y="price:Q",
        color=alt.Color("series:N", scale=alt.Scale(domain=["Actual (history)", "Actual (outcome)"],
                                                    range=["#6b7280", "#2e9e5b"]), legend=alt.Legend(title=None)),
        tooltip=tooltip)
    st.altair_chart((band + act + fc).properties(height=300), width="stretch")
    note = ("Expected price = as-of close × e^(expected log return); the band uses the 80% interval. "
            "The dashed path spreads the window's return evenly across its sessions for display only: the model "
            "predicts the window total, not a day-by-day path.")
    if gap:
        note += (f" The {gap} session(s) before the window are not forecast; prices assume no change over them, "
                 "so the real uncertainty is wider than shown.")
    st.caption(note + (" This forecast has no validated edge, so treat the price as illustrative only."
                       if p["evidence"] != "validated" else ""))


def backtest_controls(ms, key):
    fc = ms.cfg.get("forecast") or {}
    c1, c2, c3, c4 = st.columns([1, 1, 3, 1])
    h = c1.selectbox("Horizon (sessions)", fc.get("horizons", [1, 3, 5, 10, 20]), key=f"{key}_h", help=tip("horizon"))
    gap = c2.number_input("Gap (sessions)", 0, 250, 0, key=f"{key}_g", help=tip("gap"))
    models = c3.multiselect("Models", list(SPEC_BY_NAME), COMPARISON_MODELS, key=f"{key}_m",
                            format_func=lambda n: SPEC_BY_NAME[n].label, help=tip("models"))
    perms = c4.number_input("Permutations", 0, 100, 0, help=tip("permutations"),
                            key=f"{key}_p")
    tune = st.checkbox("Tune hyperparameters (slower)", key=f"{key}_tune", help=tip("tune_hyperparams"))
    return int(h), int(gap), tuple(models), int(perms), tune


def get_backtest(ms, symbol, s, px, key):
    h, gap, models, perms, tune = backtest_controls(ms, key)
    if st.button("Run backtest", type="primary", key=f"{key}_run"):
        st.session_state["bt_args"] = (symbol, s, gap, h, models, perms, tune)
    args = st.session_state.get("bt_args")
    if not args or args[0] != symbol or args[1] != s:
        st.info("Press **Run backtest** to evaluate every model walk-forward with the shared training recipe. "
                "Results are stored in the database.")
        return None, None
    symbol_, s_, gap_, h_, models_, perms_, tune_ = args
    return _backtest(ms, symbol_, s_, gap_, h_, models_, perms_, None, data_version(ms, px), tune_), args


def screen_vol(ms, symbol, s, px):
    """Volatility target: does each data category improve on the HAR volatility benchmark?"""
    from .catalog_features import CATEGORIES
    labels = {"market": "Market features (dashboard set)", **CATEGORIES}
    st.caption(tip("vol_target") or "")
    c1, c2, c3 = st.columns([1, 1, 4])
    h = c1.selectbox("Horizon (sessions)", [5, 10, 20], index=2, key="vol_h", help=tip("horizon"))
    gap = c2.number_input("Gap (sessions)", 0, 250, 0, key="vol_g", help=tip("gap"))
    groups = c3.multiselect("Data categories to add to HAR", list(labels), list(labels), key="vol_groups",
                            format_func=labels.get, help=tip("vol_groups"))
    if st.button("Run volatility backtest", type="primary", key="vol_run"):
        st.session_state["vol_args"] = (symbol, s, int(gap), int(h), tuple(groups))
    args = st.session_state.get("vol_args")
    if not args or args[0] != symbol or args[1] != s:
        ui.note("Choose a horizon and categories, then press Run volatility backtest.")
        return
    out = _vol_backtest(ms, *args, str(px.index[-1].date()))
    if out.get("status") != "ok":
        st.warning(out.get("message", "Volatility backtest unavailable."))
        return
    st.caption(f"Walk-forward evaluation {out['eval_start']} to {out['eval_end']} · horizon {args[3]} · gap {args[2]} · "
               f"{out['n']} predictions per model ({out['n_independent']} non-overlapping) · every model on the same dates")
    for n in out.get("notes", []):
        ui.note(n)
    if out.get("groups_missing"):
        ui.note("Not available for this instrument: " + ", ".join(labels.get(g, g) for g in out["groups_missing"]))
    t = out["table"].copy()
    table(st, t.round(4), hide_index=True, width="stretch")
    st.caption(f"QLIKE: standard loss for volatility forecasts (lower is better). qlike_vs_base_pct > 0 = better than HAR. "
               f"vol_up_down_accuracy: share of windows where the model correctly said whether volatility would end above "
               f"or below its current 20-day level (always guessing the more common answer scores "
               f"{out['vol_up_majority_rate']:.1%}).")
    cand = t[t.model.str.startswith("HAR +")]
    bars = alt.Chart(cand).mark_bar().encode(
        x=alt.X("qlike_vs_base_pct:Q", title="Out-of-sample QLIKE improvement over HAR (%)"),
        y=alt.Y("model:N", sort="-x", title=None),
        color=alt.condition("datum.qlike_vs_base_pct > 0", alt.value("#2563eb"), alt.value("#94a3b8")),
        tooltip=["model", alt.Tooltip("qlike_vs_base_pct:Q", format=".2f"), alt.Tooltip("dm_p_holm:Q", format=".3f")])
    st.altair_chart(bars.properties(height=28 * len(cand) + 40), width="stretch")
    sig = cand[(cand.qlike_vs_base_pct > 0) & (cand.dm_p_holm < 0.05)].sort_values("qlike")
    if len(sig):
        st.success("Significantly better than HAR after correcting for the number of comparisons: "
                   + ", ".join(sig.model.str.removeprefix("HAR + ")) + ". Re-test on new data before relying on it.")
        rev_labels = {v: k for k, v in labels.items()}
        best_group = rev_labels.get(sig.iloc[0].model.removeprefix("HAR + "))
        if best_group:
            from .forecast_service import run_vol_forecast
            live = run_vol_forecast(ms, symbol, s, args[2], args[3], best_group)
            if live is not None:
                st.info(f"**As of today ({live['as_of']:%b %d, %Y}):** expected volatility over the next "
                        f"{args[3]} sessions is **{live['expected_vol']:.1%}** (80% range "
                        f"{live['lo']:.1%}-{live['hi']:.1%}), using {labels.get(best_group, best_group)} "
                        f"- the standard HAR method alone would have said {live['baseline_vol']:.1%}.")
    else:
        st.warning("No category is significantly better than HAR after correcting for the number of comparisons "
                   "(Diebold-Mariano test with Newey-West errors, Holm adjustment).")
    heading("Predicted vs realised volatility", "vol_chart")
    ch = out["chart"].reset_index().melt("date", var_name="series", value_name="annualised volatility")
    st.altair_chart(alt.Chart(ch).mark_line(strokeWidth=1.2).encode(
        x=alt.X("date:T", title=None), y=alt.Y("annualised volatility:Q", axis=alt.Axis(format="%")),
        color="series:N"), width="stretch")


PERIODS = {"week": ("Past week", 5), "2weeks": ("Past 2 weeks", 10), "month": ("Past month", 21),
           "3months": ("Past 3 months", 60), "custom": ("Custom range", None)}


def screen_actual_vs_predicted(ms, symbol, ex, tz, s, px):
    """Left: the market data exactly as received. Right: predictions for the same sessions made without
    those sessions' prices."""
    st.caption(tip("actual_vs_predicted") or "")
    c1, c2 = st.columns([1.2, 4])
    period = c1.selectbox("Period", list(PERIODS), format_func=lambda k: PERIODS[k][0], key="avp_period",
                          help=tip("avp_period"))
    groups = c2.multiselect("Data categories", list(SELECTABLE_GROUPS), ["market"], key="avp_groups",
                            format_func=SELECTABLE_GROUPS.get, help=tip("avp_categories"))
    c3, c4 = st.columns([1.2, 4])
    algo = c3.radio("Algorithm", list(ALGORITHMS), key="avp_algo", format_func=lambda a: ALGORITHMS[a][0],
                    help=tip("avp_algorithm"))
    mode = c4.radio("Prediction method", ["strict", "rolling"], key="avp_mode", help=tip("avp_mode"),
                    format_func={"strict": "Strict: one forecast before the period",
                                 "rolling": "Rolling: each day from the day before"}.get)
    if not groups:
        ui.note("Choose at least one data category.")
        return
    model = custom_model_name(groups, algo)
    st.caption(f"Model: **{SPEC_BY_NAME[model].label}**, trained on these categories together.")
    if period == "custom":
        lo, hi = px.index[min(600, len(px) - 1)].date(), px.index[-1].date()
        st.session_state.setdefault("avp_range", (max(lo, (px.index[-1] - pd.Timedelta(days=30)).date()), hi))
        rng = st.date_input("Dates", key="avp_range", min_value=lo, max_value=hi)
        if len(rng) != 2:
            ui.note("Select both ends of the range.")
            return
        window = period_sessions(ms, symbol, start=rng[0], end=rng[1])
    else:
        window = period_sessions(ms, symbol, preset_sessions=PERIODS[period][1])
    if len(window) == 0:
        st.warning("No completed trading sessions in this period.")
        return
    if len(window) > 60:
        st.warning(f"{len(window)} sessions selected; choose at most 60.")
        return
    st.caption(f"{len(window)} completed sessions: {window[0].date()} to {window[-1].date()}"
               + (" (today's session is still open, so it is not included)" if window[-1] < px.index[-1] else ""))
    request = (symbol, s, tuple(window), model, mode)
    if st.button("Compare actual vs predicted", type="primary", key="avp_run"):
        st.session_state["avp_last"] = request
    if st.session_state.get("avp_last") != request:
        ui.note("Choose a period, model and method, then press Compare actual vs predicted.")
        return
    try:
        out = _hindcast(ms, symbol, s, tuple(window), model, mode, data_version(ms, px))
    except ValueError as e:
        st.warning(f"Prediction unavailable: {e}")
        return
    act, pr, sm = out["actual"], out["predicted"], out["summary"]
    a, b, c, d = st.columns(4)
    a.metric("Average error", f"{sm['mae_pct']:.2f}%", help=tip("avp_mae"))
    b.metric("Direction right", f"{sm['direction_hit_rate']:.0%}", help=tip("avp_direction"))
    c.metric("Actual inside 80% range", f"{sm['range_coverage']:.0%}", help=tip("avp_coverage"))
    d.metric("Forecast made after close of", sm["first_as_of"] if mode == "strict" else "each previous day")
    e, f, _, _ = st.columns(4)
    e.metric("Confident calls right", f"{sm['confident_hit_rate']:.0%}" if sm.get("confident_days") else "-",
             f"{sm.get('confident_days', 0)} of {sm['sessions']} days had a bullish/bearish call", delta_color="off",
             help=tip("avp_confident"))
    st.caption(sm.get("training", "") + " · same recipe as Future Forecast: strict day k = a forecast of the next "
               "k sessions made on the same as-of date.")
    show_range = st.toggle("Show 80% forecast range on the predicted chart", key="avp_show_range",
                           help=tip("avp_coverage"))
    lo = float(min(act["low"].min(), pr["predicted_low"].min(), pr["range_low"].min() if show_range else np.inf))
    hi = float(max(act["high"].max(), pr["predicted_high"].max(), pr["range_high"].max() if show_range else -np.inf))
    pad = 0.02 * (hi - lo or hi)
    labels = len(pr) <= 30
    domain = [lo - pad, hi + (0.35 if labels and len(pr) > 6 else 0.08 if labels else 0.02) * (hi - lo or hi)]
    # change vs the previous session's close; day 1 uses the last actual close before the period
    prev_close = act["close"].astype(float).shift(1)
    prev_close.iloc[0] = pr["reference_close"].iloc[0]
    act_chg = act.drop(columns="pct_change", errors="ignore").assign(change=act["close"] - prev_close,
                                                                      pct_change=act["close"] / prev_close - 1)
    left, right = st.columns(2)
    with left:
        heading("Actual market data", "avp_actual")
        st.caption("Exactly as received from the data provider: open, high, low, close. No forecasting. "
                   "Labels: change vs the previous close.")
        st.altair_chart(price_chart(act_chg, height=360, domain=domain, change_labels=labels), width="stretch")
    with right:
        heading("Predicted for the same dates", "avp_predicted")
        st.caption("Made without the prices of these sessions"
                   + (f" (as of the close of {sm['first_as_of']})." if mode == "strict" else " (each from the previous close).")
                   + (" Labels: predicted change vs the previous predicted close." if mode == "strict"
                      else " Labels: predicted change vs the previous actual close."))
        candles = pr[["predicted_open", "predicted_high", "predicted_low", "predicted_close"]].set_axis(
            ["open", "high", "low", "close"], axis=1)
        # predicted_open is the previous close the forecast starts from (predicted in strict, actual in rolling)
        candles = candles.assign(change=candles["close"] - candles["open"], pct_change=candles["close"] / candles["open"] - 1)
        chart = price_chart(candles, height=360, domain=domain, colors=PREDICTED_COLORS, change_labels=labels)
        if show_range:
            rng = pr[["range_low", "range_high"]].rename_axis("t").reset_index()
            band = alt.Chart(rng).mark_area(opacity=0.15, color="#64748b").encode(
                x="t:T", y=alt.Y("range_low:Q", scale=alt.Scale(zero=False, domain=domain)), y2="range_high:Q")
            chart = band + chart
        st.altair_chart(chart, width="stretch")
    heading("Day by day", "avp_table")
    t = pr.reset_index()[["date", "as_of", "sessions_ahead", "actual_close", "predicted_open", "predicted_high",
                          "predicted_low", "predicted_close", "error_pct", "range_low", "range_high", "prob_up",
                          "actual_up", "predicted_up", "confident", "direction_hit", "inside_80pct_range"]]
    t["date"], t["as_of"] = t["date"].dt.date, t["as_of"].dt.date
    table(st, t.round(4), hide_index=True, width="stretch")
    ui.note("Direction in strict mode compares each day with the close before the period; in rolling mode with the "
            "previous day's close. A short period says little about a model: use Model Comparison for "
            "statistically meaningful tests.")


def screen_models(ms, symbol, ex, tz, s, px):
    target = st.radio("Target", ["Direction (up / down)", "Volatility"], horizontal=True, key="cmp_target",
                      help=tip("cmp_target"))
    if target == "Volatility":
        screen_vol(ms, symbol, s, px)
        return
    st.caption(tip("model_comparison") or "")
    bt, args = get_backtest(ms, symbol, s, px, "cmp")
    if bt is None:
        return
    res = bt["results"]
    rows = [{"model": r["spec"].label, "family": r["spec"].family, "evidence": r["metrics"]["evidence"],
             **{k: r["metrics"].get(k) for k in ["accuracy", "accuracy_edge", "confident_accuracy", "confident_share",
                                                  "precision", "recall", "f1", "auc", "brier", "mae", "rmse",
                                                  "r2_vs_zero", "edge_p_value", "eval_start", "n", "n_independent"]}}
            for r in res.values()]
    df = pd.DataFrame(rows)
    first = next(iter(res.values()))["metrics"]
    st.caption(f"Walk-forward evaluation to {first['eval_end']} · horizon {args[3]} · gap {args[2]} · each model "
               "is scored on all of its own dates (eval_start, n), exactly as on Future Forecast and Actual vs "
               "Predicted; the head-to-head tests below use only dates both models share.")
    if bt.get("training"):
        st.caption(bt["training"])
    st.caption("accuracy_edge = accuracy minus always guessing the more common direction (above 0 = real skill). "
               "confident_accuracy = accuracy on days the model made a bullish/bearish call; confident_share = "
               "share of days with such a call.")
    for n in bt.get("notes", []):
        ui.note(n)
    if bt.get("skipped"):
        ui.note("Not run (data not available for this instrument): "
                + ", ".join(SPEC_BY_NAME[m].label for m in bt["skipped"]))
    table(st, df.round(4), hide_index=True, width="stretch")
    bars = alt.Chart(df.dropna(subset=["auc"])).mark_bar().encode(
        x=alt.X("auc:Q", scale=alt.Scale(domain=[0.4, 0.6]), title="Out-of-sample AUC (0.5 = coin flip)"),
        y=alt.Y("model:N", sort="-x", title=None), color="family:N",
        tooltip=["model", alt.Tooltip("auc:Q", format=".4f"), alt.Tooltip("accuracy:Q", format=".2%")])
    rule = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart((bars + rule).properties(height=36 * len(df) + 40), width="stretch")
    heading("Does astrology add predictive value?", "astro_value")
    allvs = pd.DataFrame(bt["astro_vs"])
    fam = {n: SPEC_BY_NAME[n].family for n in allvs.model} if len(allvs) else {}
    ext = allvs.model.map(fam).isin(["category", "geopolitics", "weather"]) if len(allvs) else pd.Series(dtype=bool)
    vs = allvs[~ext] if len(allvs) else allvs
    if len(vs):
        table(st, vs.round(4), hide_index=True, width="stretch")
        better = vs[(vs.model.str.startswith(("astro", "hybrid"))) & (vs.accuracy_diff > 0) & (vs.mcnemar_p < 0.05)]
        if len(better):
            st.success("At least one astro/hybrid model is significantly more accurate than its comparison "
                       "model on these dates. Re-test on new data before trusting it.")
        else:
            st.warning("No astro or hybrid model is significantly more accurate than its non-astro comparison "
                       "(exact McNemar test on non-overlapping windows). On this evidence astrology adds no "
                       "measurable predictive value for this instrument and horizon.")
    ev = allvs[ext] if len(allvs) else allvs
    if len(ev):
        heading("Does each data category add predictive value?", "external_value")
        st.caption("Each 'market + category' model is compared with the market-only model of the same type, on "
                   "identical dates. With many categories some will look better by chance: the Holm column "
                   "corrects for the number of comparisons.")
        from .risk_eval import holm
        ev = ev.assign(category=[SPEC_BY_NAME[n].label.removeprefix("Market + ") for n in ev.model],
                       mcnemar_p_holm=holm(ev.mcnemar_p.fillna(1.0)))
        cols = ["category", "vs", "accuracy_diff", "auc_diff", "mcnemar_p", "mcnemar_p_holm", "rmse_diff",
                "n_independent"]
        table(st, ev[cols].round(4), hide_index=True, width="stretch")
        better = ev[(ev.accuracy_diff > 0) & (ev.mcnemar_p_holm < 0.05)]
        if len(better):
            st.success(f"{len(better)} category model(s) are significantly more accurate than market-only after "
                       "correcting for multiple comparisons. Re-test on new data before trusting them.")
        else:
            st.warning("No category model is significantly more accurate than the market-only model after "
                       "correcting for the number of comparisons (exact McNemar test, Holm adjustment).")
    if bt.get("coverage") is not None and len(bt["coverage"]):
        heading("Data coverage and lineage (categories 00 and 22)", "data_coverage")
        st.caption("Which catalog categories were built for this instrument, from when, and how complete they are. "
                   "Categories without a point-in-time data source are listed with the reason; they are never filled in.")
        table(st, bt["coverage"], hide_index=True, width="stretch")
    if "permutation" in bt:
        pm = bt["permutation"]
        st.markdown(f"**Permutation test (Model A):** real AUC {pm['real_auc']:.4f} vs shuffled mean "
                    f"{pm['null_mean']:.4f}, p = {pm['p_value']:.3f}")
        st.altair_chart(alt.Chart(pd.DataFrame({"auc": pm["null"]})).mark_bar().encode(
            x=alt.X("auc:Q", bin=alt.Bin(maxbins=20)), y="count()").properties(height=160) +
            alt.Chart(pd.DataFrame({"x": [pm["real_auc"]]})).mark_rule(color="red").encode(x="x:Q"),
            width="stretch")


def screen_backtests(ms, symbol, ex, tz, s, px):
    bt, args = get_backtest(ms, symbol, s, px, "bt")
    if bt is not None:
        res = bt["results"]
        from .forecast_engine import strategy
        shared = common_dates(res)                 # growth curves start on the same date for every model
        st.caption(f"Strategy curves use the {len(shared)} dates every selected model has a prediction for "
                   f"({shared.min():%Y-%m-%d} to {shared.max():%Y-%m-%d})." if len(shared) else
                   "The selected models share no prediction dates.")
        cost = total_cost_bps(ms.cfg)
        curves, rows = [], []
        for name, r in res.items():
            if name == "naive_zero":
                continue
            eq, summ = strategy(r["oof"].loc[shared] if len(shared) else r["oof"], args[3], cost)
            curves.append(pd.DataFrame({"date": eq.index, "model": r["spec"].label,
                                        "growth": np.exp(eq["strategy"].cumsum())}))
            rows.append({"model": r["spec"].label, **{f"strategy_{k}": v for k, v in summ["strategy"].items()},
                         **{f"buy_hold_{k}": v for k, v in summ["buy_hold"].items()}})
        curves.append(pd.DataFrame({"date": eq.index, "model": "Buy & hold", "growth": np.exp(eq["buy_hold"].cumsum())}))
        cv = pd.concat(curves)
        st.altair_chart(alt.Chart(cv).mark_line().encode(
            x=alt.X("date:T", title=None), y=alt.Y("growth:Q", scale=alt.Scale(type="log"), title="Growth of 1 (log)"),
            color="model:N", tooltip=["date:T", "model", alt.Tooltip("growth:Q", format=".3f")])
            .properties(height=320).interactive(bind_y=False), width="stretch")
        heading("Strategy simulation", "strategy_simulation")
        st.caption(f"Long-or-cash simulation, one decision per {args[3]}-session window, "
                   f"{cost:.0f} bps cost+slippage per position change. Simulation only: no orders are placed.")
        table(st, pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
        pick = st.selectbox("Predicted vs actual", [n for n in res if n != "naive_zero"],
                            format_func=lambda n: SPEC_BY_NAME[n].label, help=tip("predicted_vs_actual"))
        o = res[pick]["oof"].reset_index()
        st.altair_chart(alt.Chart(o.sample(min(len(o), 3000), random_state=0)).mark_circle(size=12, opacity=.4)
                        .encode(x=alt.X("expected:Q", title="Predicted log return"),
                                y=alt.Y("actual:Q", title="Actual log return"),
                                tooltip=["date:T", alt.Tooltip("expected:Q", format="+.4f"),
                                         alt.Tooltip("actual:Q", format="+.4f")]).properties(height=280),
                        width="stretch")
        roll = ((o.p_up > 0.5) == (o.actual > 0)).rolling(250).mean()
        st.line_chart(pd.DataFrame({"rolling 250-prediction accuracy": roll.to_numpy(),
                                    "share of up windows": (o.actual > 0).rolling(250).mean().to_numpy()},
                                   index=o.date), height=200)

    heading("Stored backtest runs", "stored_backtests")
    runs = repo.list_backtests(ms.con, symbol)
    keep = [c for c in ["created_at", "model", "version", "horizon_sessions", "gap_sessions", "zodiac", "eval_start",
                        "eval_end", "accuracy", "auc", "f1", "rmse", "edge_p_value", "evidence",
                        "backtest.strategy.sharpe", "backtest.strategy.max_drawdown", "backtest.buy_hold.sharpe"]
            if c in runs.columns]
    table(st, runs[keep].round(4) if len(runs) else runs, hide_index=True, width="stretch", height=240)
    heading("Stored forecasts and realised outcomes", "stored_forecasts")
    if st.button("Update realised returns", help=tip("update_realised")):
        n = refresh_realized(ms, symbol)
        st.toast(f"Updated {n} forecast(s)")
    preds = repo.list_predictions(ms.con, symbol)
    if len(preds):
        done = preds.dropna(subset=["realized_return"])
        done = done[done.direction != "neutral"]
        if len(done):
            hit = ((done.realized_return > 0) == (done.direction == "bullish")).mean()
            st.caption(f"{len(done)} matured directional forecasts: {hit:.0%} correct. Small samples are "
                       "dominated by chance; replayed forecasts are point-in-time but were chosen by the user.")
    table(st, preds.round(4), hide_index=True, width="stretch", height=260)


def screen_glossary(ms):
    st.caption("Plain-language explanations of the market, data-category, astrology and statistics terms used in this app. "
               "The same texts appear as tooltips (?) next to controls, metrics and table columns.")
    g = repo.list_help(ms.con)
    c1, c2 = st.columns([2, 1])
    q = c1.text_input("Search terms", placeholder="e.g. nakshatra, AUC, drawdown")
    cats = c2.multiselect("Category", sorted(g.category.unique()))
    if q:
        g = g[g.term.str.contains(q, case=False, regex=False) | g.description.str.contains(q, case=False, regex=False)
              | g.key.str.contains(q, case=False, regex=False)]
    if cats:
        g = g[g.category.isin(cats)]
    if not st.toggle("Edit help texts", help="Changes are saved to the help_texts table and used by every tooltip."):
        for cat, grp in g.groupby("category"):
            with st.expander(f"{cat.title()} ({len(grp)})", expanded=bool(q)):
                for r in grp.itertuples():
                    st.markdown(f"**{r.term}**  \n{r.description}")
        return
    st.caption("`key` is the control id or table column name the tooltip attaches to (lower-case). "
               "Add rows to cover new terms; delete a row to remove its tooltip.")
    edited = st.data_editor(g[["key", "term", "category", "description"]], num_rows="dynamic", hide_index=True,
                            width="stretch", key="help_editor",
                            column_config={"description": st.column_config.TextColumn(width="large")})
    if st.button("Save help texts", type="primary"):
        removed = set(g.key) - set(edited.key.dropna().str.strip().str.lower())
        n = repo.save_help(ms.con, edited)
        repo.delete_help(ms.con, removed)
        st.toast(f"Saved {n} help text(s)" + (f", removed {len(removed)}" if removed else ""))
        st.rerun()


def render(cfg):
    """App shell: sidebar navigation, shared top bar, then the selected page."""
    from . import ui_dashboard as dash
    ui.inject_css()
    dash.keep_state()
    ms = _service(cfg, cfg.get("database", {}).get("path"))
    _HELP.clear()
    _HELP.update(repo.help_texts(ms.con))
    ctx = {"cfg": cfg, "ms": ms}
    anchor = (cfg.get("forecast") or {}).get("session_anchor", "open")

    def page(fn, title):
        def run():
            if title != "Home":
                st.subheader(title)
            fn(ms, ctx["symbol"], ctx["ex"], ctx["tz"], ctx["s"], ctx["px"])
        run.__name__ = title.lower().replace(" ", "_")
        return run

    pages = {
        "home": st.Page(lambda: dash.home(ctx, pages), title="Home", icon=":material/home:", default=True),
        "overview": st.Page(page(screen_overview, "Market Overview"), title="Market Overview",
                            icon=":material/bar_chart:", url_path="overview"),
        "history": st.Page(page(lambda *a: screen_history(*a, anchor), "Historical Analysis"),
                           title="Historical Analysis", icon=":material/history:", url_path="history"),
        "forecast": st.Page(page(screen_forecast, "Future Forecast"), title="Future Forecast",
                            icon=":material/calendar_month:", url_path="forecast"),
        "models": st.Page(page(screen_models, "Model Comparison"), title="Model Comparison",
                          icon=":material/account_tree:", url_path="models"),
        "backtests": st.Page(page(screen_backtests, "Backtesting Results"), title="Backtesting Results",
                             icon=":material/description:", url_path="backtests"),
        "actual_vs_predicted": st.Page(page(screen_actual_vs_predicted, "Actual vs Predicted"),
                                       title="Actual vs Predicted", icon=":material/compare:",
                                       url_path="actual-vs-predicted"),
    }
    glossary = st.Page(lambda: (st.subheader("Glossary"), screen_glossary(ms)), title="Glossary",
                       icon=":material/menu_book:", url_path="glossary")
    nav = st.navigation({"Market Prediction": list(pages.values()), "More": [glossary]})
    # full logo (white text) suits the navy sidebar; the light top bar of a collapsed sidebar gets the icon
    st.logo(str(LOGO), size="large", icon_image=str(LOGO.with_name("logo_icon.svg")))
    st.sidebar.html(f'<div class="am-disclaimer">ⓘ {tip("disclaimer_short") or DISCLAIMER}</div>')
    with st.sidebar.expander("Data & storage"):
        st.caption(ms.provider.notes)
        st.json(repo.table_counts(ms.con), expanded=False)
    if nav in pages.values():
        ctx["page"] = nav
        dash.top_bar(ctx, home_page=pages["home"])
        ctx["px"] = load_prices(ms, ctx["symbol"], ctx["ex"])
    nav.run()
