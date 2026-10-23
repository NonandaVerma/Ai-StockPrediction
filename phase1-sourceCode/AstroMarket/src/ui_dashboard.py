"""Beginner-friendly dashboard: shared top bar + Home page.

Every number shown comes from the existing services (MarketService quotes/bars, run_forecast,
astrology.chart); this module only arranges and words them.
"""
import re

import numpy as np
import pandas as pd
import streamlit as st

from . import repository as repo
from . import ui_components as ui
from .astrology import _asp_label, chart, readable
from .ephemeris import AYANAMSAS, AstroSettings
from .exchanges import EXCHANGES, exchange_for_symbol, session_time_utc
from .forecast_models import DEFAULT_MODELS, SPEC_BY_NAME
from .forecast_service import run_vol_forecast
from .market_providers import ProviderError
from .ui_forecast import (PRESETS, _forecast, _intraday, _latest, data_version, st_searchbox, suggest_symbols, tip)

TARGETS = {"next1": "Next trading day", "next3": "Next 3 trading days", "next5": "Next 5 trading days",
           "next10": "Next 10 trading days", "next20": "Next 20 trading days",
           "date": "Pick a date...", "range": "Pick a date range..."}
TZ_LABELS = {"Asia/Kolkata": "IST (Asia/Kolkata)", "America/New_York": "ET (America/New_York)",
             "Europe/London": "UK (Europe/London)", "UTC": "UTC"}
SYSTEMS = {"sidereal": "Vedic (Indian)", "tropical": "Western (Tropical)"}
HEADLINE = ["astro_stat", "astro_lgbm", "hybrid_lgbm", "market_linear"]
RISK = {"low": "Low", "normal": "Moderate", "elevated": "Elevated", "high": "High"}
DIRECTION = {"bullish": ("Upward", "Bullish", "up", "↗"), "bearish": ("Downward", "Bearish", "down", "↘"),
             "neutral": ("Sideways", "Neutral", "neutral", "→")}
PERIODS = {"1D": ("5m", 5, 1), "1W": ("30m", 10, 5), "1M": 31, "3M": 92, "6M": 183, "1Y": 366}
# widget keys kept alive while the user is on pages that do not show the top bar (e.g. Glossary)
PERSIST = ["am_target", "am_date", "am_range", "am_tz", "am_system", "am_exchange", "am_ayanamsa", "am_node",
           "am_orb", "am_model", "am_period"]


def keep_state():
    for k in PERSIST:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]


# ---------------------------------------------------------------- shared header + top bar
def _toggle_settings():
    st.session_state["am_settings_open"] = not st.session_state.get("am_settings_open", False)


def top_bar(ctx: dict, home_page=None) -> None:
    """Header, filter bar and right-hand settings panel. Fills ctx with symbol, ex, tz, s, req."""
    cfg, ms = ctx["cfg"], ctx["ms"]
    for k, v in {"am_target": "next5", "am_ayanamsa": cfg["astro"]["ayanamsa"], "am_period": "1M",
                 "am_orb": float(cfg["astro"]["aspect_orb_deg"]), "am_system": cfg["astro"]["zodiac"]}.items():
        st.session_state.setdefault(k, v)                 # defaults live in session state (see keep_state)
    st.session_state.setdefault("am_settings_open", False)
    slots = {}
    ui.page_header("Market Prediction",
                   "Forecast market moves from price, macro, global, geopolitical, weather and astrological data — past, present and future.",
                   right=lambda mid, gear: slots.update(clock=mid.empty(), gear=gear))
    slots["gear"].button("", icon=":material/settings:", key="am_settings_toggle", help=tip("settings"),
                         type="primary" if st.session_state["am_settings_open"] else "secondary",
                         on_click=_toggle_settings)
    # Right-hand settings panel. Its widgets are rendered even while it is hidden (slid off-screen),
    # so their values keep driving every page.
    ui.side_panel_css(st.session_state["am_settings_open"])
    panel = st.container(key="am_settings_panel")
    # Forecast Date + Generate Forecast only drive the Home forecast card, so other pages show just the stock
    on_home = home_page is None or ctx.get("page") is home_page
    with st.container(border=True):
        if on_home:
            c1, c2, c5 = st.columns([3, 2.4, 1.6], vertical_alignment="bottom")
        else:
            c1, _ = st.columns([3, 4], vertical_alignment="bottom")
        symbol = stock_picker(c1, ms, cfg)
        ex_guess, guessed = exchange_for_symbol(symbol)
        with panel:
            t, x = st.columns([4, 1], vertical_alignment="center")
            t.markdown("**Settings**")
            x.button("", icon=":material/keyboard_double_arrow_right:", key="am_settings_close",
                     help="Hide settings", on_click=_toggle_settings)
            codes = list(EXCHANGES)
            if st.session_state.get("am_exchange_for") != symbol:      # re-detect when the stock changes
                st.session_state["am_exchange"] = ex_guess.code
                st.session_state["am_exchange_for"] = symbol
            ex = EXCHANGES[st.selectbox("Exchange", codes, key="am_exchange", format_func=lambda c: EXCHANGES[c].label,
                                        help=(tip("exchange") or "") + ("\n\n**Guessed from the symbol: please confirm.**"
                                                                        if guessed else ""))]
            tzs = list(dict.fromkeys([ex.tz, "Asia/Kolkata", "America/New_York", "Europe/London", "UTC"]))
            tz = st.selectbox("Timezone", tzs, key="am_tz", format_func=lambda z: TZ_LABELS.get(z, z),
                              help=tip("display_timezone"))
            zodiac = st.selectbox("Astrology System", list(SYSTEMS), format_func=SYSTEMS.get, key="am_system",
                                  help=tip("astro_system"))
            ay = st.selectbox("Ayanamsa", list(AYANAMSAS), key="am_ayanamsa", help=tip("ayanamsa"),
                              disabled=zodiac == "tropical")
            node = st.radio("Lunar node", ["mean", "true"], horizontal=True, key="am_node", help=tip("lunar_node"))
            orb = st.slider("Aspect orb (deg)", 1.0, 10.0, step=0.5, key="am_orb", help=tip("aspect_orb"))
            primary = st.selectbox("Forecast model", HEADLINE, key="am_model", help=tip("headline_model"),
                                   format_func=lambda n: SPEC_BY_NAME[n].label)
        go = date = start = end = None
        if on_home:
            target = c2.selectbox("Forecast Date", list(TARGETS), format_func=TARGETS.get, key="am_target",
                                  help=tip("forecast_date"))
            go = c5.button("Generate Forecast", type="primary", icon=":material/auto_awesome:", width="stretch",
                           help=tip("generate_forecast"))
            today = pd.Timestamp.now(tz=tz).date()
            if target == "date":
                st.session_state.setdefault("am_date", today + pd.Timedelta(days=7))
                date = c2.date_input("Target date", key="am_date",
                                     label_visibility="collapsed")
            elif target == "range":
                st.session_state.setdefault("am_range", (today + pd.Timedelta(days=1), today + pd.Timedelta(days=14)))
                r = c2.date_input("Target range", key="am_range", label_visibility="collapsed")
                start, end = (r if len(r) == 2 else (None, None))
    slots["clock"].html(f'<div class="am-clock">📅 {pd.Timestamp.now(tz=tz):%b %d, %Y &nbsp; %I:%M %p} '
                        f'({TZ_LABELS.get(tz, tz).split(" ")[0]})</div>')
    s = AstroSettings.from_cfg(cfg, zodiac=zodiac, ayanamsa=ay, node=node, orb=orb)
    ctx.update(symbol=symbol, ex=ex, tz=tz, s=s)
    if go:
        if target == "range" and start is None:
            st.warning("Select both ends of the date range.")
        else:
            st.session_state["am_request"] = {
                "symbol": symbol, "s": s, "kind": "next" if target.startswith("next") else target,
                "horizon": int(target[4:]) if target.startswith("next") else 1, "date": date, "start": start,
                "end": end, "primary": primary, "models": tuple(dict.fromkeys([primary, *DEFAULT_MODELS]))}
    req = st.session_state.get("am_request")
    ctx["req"] = req if req and req["symbol"] == symbol and req["s"] == s else None
    ctx["stale_request"] = bool(req) and ctx["req"] is None


def stock_picker(where, ms, cfg) -> str:
    known = {**PRESETS, **{r.symbol: r.name if pd.notna(r.name) and r.name else r.symbol    # missing name -> NaN
                           for r in repo.list_instruments(ms.con).itertuples()}}
    known.update(st.session_state.get("symbol_names", {}))
    current = st.session_state.setdefault("am_symbol", cfg["market"].get("ticker", "^NSEI"))
    name = str(known.get(current, ""))
    label = f"{current} · {name[:13] + '…' if len(name) > 14 else name}" if name else current

    def search(term):
        hits = suggest_symbols(ms, known, term)
        t = term.strip().upper()
        if re.fullmatch(r"[\^A-Z0-9.\-=]{1,15}", t) and t not in {h[1] for h in hits}:
            hits.append((f"Use “{t}” as typed", t))
        return hits

    with where:
        if st_searchbox is not None:
            # fixed-height box + absolutely positioned iframe (CSS .st-key-am_symbol_box): the suggestion
            # list floats over the page instead of stretching the filter bar
            with st.container(key="am_symbol_box"):
                picked = st_searchbox(search, key="am_symbol_search", label="Stock / Index", placeholder=label,
                                      default_options=[(f"{k} · {v}", k) for k, v in list(known.items())[:12]],
                                      debounce=300, help=tip("stock_index"))
        else:
            picked = st.text_input("Stock / Index", placeholder=label, help=tip("stock_index")).strip().upper() or None
    if picked and picked != current:
        st.session_state["am_symbol"] = picked
        st.rerun()
    return current


def stock_name(ms, symbol: str) -> str:
    inst = ms.instrument(symbol) or {}
    return inst.get("name") or st.session_state.get("symbol_names", {}).get(symbol) or PRESETS.get(symbol) or symbol


# ---------------------------------------------------------------- Home page
def home(ctx: dict, pages: dict):
    ms, symbol, ex, tz, s, px = (ctx[k] for k in ("ms", "symbol", "ex", "tz", "s", "px"))
    o = None
    left, right = st.columns([1.55, 1])
    with right, st.container(border=True):           # forecast first, so its spinner sits inside this card
        if ctx["req"]:
            r = ctx["req"]
            try:
                o = _forecast(ms, symbol, s, px.index[-1], r["kind"], r["horizon"], r["date"], r["start"],
                              r["end"], r["models"], r["primary"], data_version(ms, px))
            except ValueError as e:
                st.error(str(e))
        forecast_card(ctx, o, pages)
    with left, st.container(border=True):
        price_card(ms, symbol, ex, tz, px, o)
    # astrology is one data category among several: its three cards sit under their own heading
    st.markdown("##### :material/category: Category: Astrology", help=tip("astro_category"))
    a, b, c = st.columns(3)
    with a, st.container(border=True):
        planets_card(s)
    with b, st.container(border=True):
        similar_dates_card(o)
    with c, st.container(border=True):
        meaning_card(o, s)


def price_card(ms, symbol, ex, tz, px, o):
    st.markdown(f"#### {stock_name(ms, symbol)} ({ex.code}) – Price & Forecast", help=tip("price_forecast_chart"))
    top, per = st.columns([1, 1.3], vertical_alignment="center")
    try:
        q = _latest(ms, symbol)
        chg = q["price"] - q["price"] / (1 + q["change_pct"])
        when = q["ts_utc"].tz_convert(tz)
        ui.price_header(q["price"], chg, q["change_pct"], ex.currency_symbol,
                        f"{q['label'].title()} ({ex.code}) • {when:%b %d, %I:%M %p} {TZ_LABELS.get(tz, tz).split(' ')[0]}"
                        f" • {'market open' if q['market_open'] else 'market closed'}", where=top)
    except ProviderError as e:
        last, prev = px["close"].iloc[-1], px["close"].iloc[-2]
        ui.price_header(last, last - prev, last / prev - 1, ex.currency_symbol,
                        f"Last close {px.index[-1]:%b %d, %Y} (live quote unavailable: {e})", where=top)
    period = per.segmented_control("Period", list(PERIODS), key="am_period",
                                   label_visibility="collapsed", help=tip("period")) or "1M"
    bars, intraday_note = chart_bars(ms, symbol, tz, px, period)
    fc = None
    if o and o.get("status") == "ok":
        p, w = o["primary"], o["window"]
        times = session_time_utc(ex, [o["as_of"], *w], "close").tz_convert(tz).tz_localize(None)
        fc = ui.growth_path(o["last_close"], p["expected_return"], p["interval_low"], p["interval_high"], list(times))
        if o["gap"]:
            intraday_note += f" The {o['gap']} session(s) before the target window are not forecast."
    st.altair_chart(ui.candles_with_forecast(bars, fc, split_labels=(
        "Actual", f"Forecast (next {len(o['window'])} sessions)" if fc is not None else "")), width="stretch")
    ui.legend([("Actual price (up / down)", ui.UP), ("Expected path", ui.ACCENT),
               ("80% forecast range", ui.tint(ui.ACCENT, .25))] if fc is not None else
              [("Actual price (up / down)", ui.UP)])
    ui.note(intraday_note + (" The dashed path spreads the window's expected return evenly across its sessions for "
                             "display; the model predicts the window total, not each day." if fc is not None else ""))


def chart_bars(ms, symbol, tz, px, period) -> tuple[pd.DataFrame, str]:
    """OHLC bars indexed by display-time timestamps."""
    spec = PERIODS[period]
    if isinstance(spec, tuple):
        interval, days, sessions = spec
        try:
            intr = _intraday(ms, symbol, interval, days)
            keep = intr.index.normalize().unique()[-sessions:]
            d = intr[intr.index.normalize().isin(keep)].copy()
            d.index = d["ts_utc"].dt.tz_convert(tz).dt.tz_localize(None)
            return d, f"{interval} bars, times in {tz}; free feeds are delayed."
        except ProviderError as e:
            note = f"Intraday data unavailable ({e}); showing daily bars. "
            spec = 31
    else:
        note = ""
    d = px[px.index >= px.index[-1] - pd.Timedelta(days=spec)].copy()
    d.index = d["ts_utc"].dt.tz_convert(tz).dt.tz_localize(None)
    return d, note + "Daily bars (one per completed trading session)."


def _vol_tile_forecast(ms, symbol, s, gap, horizon, exchange_code):
    """Live volatility forecast from the HAR+benchmark/VIX model (src/vol_engine.py), for the
    Expected Volatility tile. Restricted to NSE/BSE: the significance + locked-holdout testing
    behind the "validated" claim this tile makes was only ever run on Indian symbols (where
    category 05 also includes India VIX, not just benchmark correlation) - the category exists for
    every exchange, but showing a "validated" claim for an untested market would overstate the
    evidence. None here falls back to the tile's existing plain-language estimate only."""
    from .catalog_features import INDIA
    if exchange_code not in INDIA:
        return None
    try:
        return run_vol_forecast(ms, symbol, s, gap, horizon)
    except Exception:
        return None


def forecast_card(ctx, o, pages):
    r = ctx["req"]
    title = "Forecast"
    if r:
        title = {"next": f"{r['horizon']}-Day Forecast (Next Trading Sessions)",
                 "date": f"Forecast for {r['date']:%b %d, %Y}" if r["date"] else "Forecast",
                 "range": "Forecast for the Selected Range"}[r["kind"]]
    h, b = st.columns([3, 1], vertical_alignment="center")
    h.markdown(f"#### :material/track_changes: {title}", help=tip("forecast_summary"))
    if o is None:
        if ctx["stale_request"]:
            st.info("The stock or settings changed. Press **Generate Forecast** to update.")
        else:
            st.info("Choose a stock and a forecast date, then press **Generate Forecast**. Each model is "
                    "re-checked against past data first, so this can take up to a minute.")
        return
    if o["status"] != "ok":
        st.error(f"Forecast unavailable. {o['message']}")
        return
    p, ex = o["primary"], ctx["ex"]
    word, badge, tone, arrow = DIRECTION[p["direction"]]
    ui.pill(badge, tone, where=b)
    c1, c2 = st.columns(2)
    ui.tile("Expected Direction", f"{arrow} {word}", f"chance of a rise: {p['prob_up']:.0%}", tone=tone,
            help=tip("direction"), where=c1)
    base = o["last_close"]
    lo, hi = base * np.exp(p["interval_low"]), base * np.exp(p["interval_high"])
    ui.tile("Estimated Return", f"{np.expm1(p['expected_return']):+.1%}",
            f"80% range {np.expm1(p['interval_low']):+.1%} to {np.expm1(p['interval_high']):+.1%} "
            f"({ex.currency_symbol}{lo:,.0f} – {ex.currency_symbol}{hi:,.0f})", tone="accent",
            help=tip("expected_return"), where=c2)
    c1, c2 = st.columns(2)
    vol_sub = f"typical swing ±{o['expected_vol']:.1%} over the window"
    vf = _vol_tile_forecast(ctx["ms"], ctx["symbol"], ctx["s"], o["gap"], o["horizon"], ctx["ex"].code)
    if vf is not None:
        vol_sub += (f"  ·  validated model: {vf['expected_vol']:.1%} "
                    f"(standard method said {vf['baseline_vol']:.1%} - this model tested ~20% more "
                    f"accurate on past data, confirmed on data it never trained on)")
    ui.tile("Expected Volatility", RISK.get(o["risk"], o["risk"].title()), vol_sub,
            help=tip("expected_volatility"), where=c1)
    conf = p["confidence"]
    if conf.get("calibrated"):
        ui.tile("Confidence", f"{conf['hit_rate']:.0%}", f"right this often in {conf['n']} past tests",
                bar=conf["hit_rate"], bar_tone="up" if conf["hit_rate"] >= .5 else "down",
                help=tip("validated_hit_rate"), where=c2)
    else:
        ui.tile("Confidence", "Not yet proven", "too few similar past tests to measure", bar=0.0,
                help=tip("confidence"), where=c2)
    w = o["window"]
    c1, c2 = st.columns(2)
    ui.tile("Target Period", f"{w[0]:%b %d, %Y}" + (f" – {w[-1]:%b %d, %Y}" if len(w) > 1 else ""),
            f"{len(w)} trading session(s)", where=c1)
    ui.tile("Prediction Generated", f"{o['generated_at'].tz_convert(ctx['tz']):%b %d, %Y %I:%M %p}",
            f"using prices up to {o['as_of']:%b %d, %Y}", where=c2)
    st.write("")
    ui.bullet_box("Key Reasons", key_reasons(o), where=st)
    for n in o["notes"]:
        ui.note(n)
    if st.button("See full forecast details", icon=":material/arrow_forward:", type="tertiary"):
        st.switch_page(pages["forecast"])


def planets_card(s):
    st.markdown("#### :material/public: Current Planetary Positions", help=tip("current_positions"))
    c = chart(pd.Timestamp.now(tz="UTC"), s)
    lun = c["lunar"]
    ui.planet_rows(c["positions"], ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn"],
                   extra=[(moon_icon(lun["elongation"]), "Moon Phase", f"{lun['phase']} ({lun['illumination']:.0%})")])
    ui.note(f"{SYSTEMS[s.zodiac]} zodiac" + (f", {s.ayanamsa} ayanamsa" if s.zodiac == "sidereal" else "")
            + ". ℞ = retrograde.")


def similar_dates_card(o):
    h, link = st.columns([3, 1], vertical_alignment="center")
    h.markdown("#### :material/history: Past Similar Dates", help=tip("past_similar_dates"))
    an = (o or {}).get("analogues") if o and o.get("status") == "ok" else None
    if an is None:
        st.caption("Generate a forecast to find past dates whose planetary positions resembled the target period.")
        return
    if an["status"] != "ok":
        st.warning(an["message"])
        return
    t = an["table"].assign(ret=np.expm1(an["table"].forward_return))
    with link.popover("View all", type="tertiary"):
        st.dataframe(t[["date", "ret", "distance"]].assign(date=t.date.dt.date).rename(
            columns={"ret": "return"}), hide_index=True, column_config={
            "return": st.column_config.NumberColumn(format="percent"),
            "distance": st.column_config.NumberColumn(format="%.2f", help=tip("distance"))})
    st.caption("These past dates had the most similar planetary positions:")
    ui.outcome_rows(t.head(5), f"Return (next {o['horizon']} days)")
    ui.note("Based on historical data and similar planetary configurations. Past results do not guarantee "
            "future performance.")


def meaning_card(o, s):
    st.markdown("#### :material/menu_book: What This Means", help=tip("what_this_means"))
    if o and o.get("status") == "ok":
        ui.explain_card("Planetary Alignment", window_summary(o, s), "✦", "accent")
        an = o["analogues"]
        if an["status"] == "ok":
            ui.explain_card("Historical Pattern",
                            f"After the {an['n']} most similar past skies the price rose {an['hit_rate']:.0%} of the "
                            f"time, versus {an['base_hit_rate']:.0%} on all days. This difference is "
                            f"{an['evidence']}.", "⇄", "warn")
    else:
        st.caption("Generate a forecast to see the planetary picture for the target period and how similar skies "
                   "played out before.")
    ui.explain_card("Keep in Mind", tip("keep_in_mind") or "", "ℹ", "violet")


# ---------------------------------------------------------------- wording built from real outputs
def feature_label(f: str) -> str:
    if f.startswith("asp_"):
        return _asp_label(f)
    m = re.fullmatch(r"([a-z]+)_(sin|cos)", f)
    if m and m.group(1) != "phase":
        return f"{m.group(1).title()}'s zodiac position"
    if f.endswith("_retro"):
        return f"{f[:-6].title()} retrograde"
    if f.startswith("moon_phase"):
        return "Moon phase"
    if f == "waxing":
        return "Waxing vs waning Moon"
    return f"market indicator '{f}'"


def key_reasons(o) -> list[str]:
    p = o["primary"]
    out = [f"{p['label']} gives a {p['prob_up']:.0%} chance that the price ends the period higher."]
    if p["direction"] != "neutral" and (p["expected_return"] > 0) != (p["direction"] == "bullish"):
        out.append("The direction call and the return estimate point different ways (they are estimated "
                   "separately), which is a sign the signal is weak.")
    feats = list(dict.fromkeys(feature_label(f) for f in p.get("selected_features") or []))
    if feats:
        out.append("Factors the model leaned on most: " + ", ".join(feats[:3]) + ".")
    elif p["model"].startswith("astro"):
        out.append("No astrological factor showed a consistent link with past prices, so the forecast stays close "
                   "to this stock's long-run average.")
    an = o.get("analogues", {})
    if an.get("status") == "ok":
        out.append(f"Similar past skies were followed by a rise {an['hit_rate']:.0%} of the time "
                   f"(all days: {an['base_hit_rate']:.0%}).")
    out.append(f"Recent price swings are {RISK.get(o['risk'], o['risk']).lower()} for this stock.")
    out.append("Past testing showed a real edge over simple guessing." if p["evidence"] == "validated" else
               "Past testing did not show a reliable edge over simple guessing, so treat this as experimental.")
    return out


def window_summary(o, s) -> str:
    r = readable(o["window_sky"], s.zodiac)
    first, last = r.iloc[0], r.iloc[-1]
    moon = first["moon_sign"] if first["moon_sign"] == last["moon_sign"] else \
        f"moves from {first['moon_sign']} to {last['moon_sign']}"
    retro = sorted({x.strip() for v in r["retrograde"] for x in str(v).split(",") if x.strip() not in ("", "-")})
    asp = [a for a in str(first["major_aspects"]).split(", ") if a and a != "-"]
    txt = f"During the target period the Sun is in {first['sun_sign']} and the Moon " + \
          (f"is in {moon}" if first["moon_sign"] == last["moon_sign"] else moon) + "."
    txt += f" Retrograde: {', '.join(retro)}." if retro else " No planet is retrograde."
    if asp:
        txt += f" Main aspects: {', '.join(asp[:3])}."
    return txt


def moon_icon(elongation: float) -> str:
    return "🌑🌒🌓🌔🌕🌖🌗🌘"[int(((elongation % 360) + 22.5) // 45) % 8]
