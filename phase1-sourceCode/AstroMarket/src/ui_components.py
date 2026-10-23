"""Reusable presentation components for the Streamlit UI (no data access, no calculations).

Colours use semi-transparent tints and inherit the text colour, so every component reads in both
the light and the dark Streamlit theme.
"""
from html import escape

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

UP, DOWN, ACCENT, NEUTRAL = "#16a34a", "#dc2626", "#2563eb", "#64748b"
TONES = {"up": UP, "down": DOWN, "accent": ACCENT, "neutral": NEUTRAL, "warn": "#d97706", "violet": "#7c3aed"}
PLANET_GLYPHS = {"sun": ("☉", "#f59e0b"), "moon": ("☽", "#6366f1"), "mercury": ("☿", "#10b981"),
                 "venus": ("♀", "#ec4899"), "mars": ("♂", "#ef4444"), "jupiter": ("♃", "#eab308"),
                 "saturn": ("♄", "#64748b"), "uranus": ("♅", "#06b6d4"), "neptune": ("♆", "#3b82f6"),
                 "pluto": ("♇", "#78716c"), "rahu": ("☊", "#8b5cf6"), "ketu": ("☋", "#a855f7")}

CSS = """
<style>
/* Streamlit's fixed top bar is 3.75rem tall: start content below it so the page header is never covered */
[data-testid="stMainBlockContainer"], .block-container {padding-top: 4.75rem; padding-bottom: 2rem; max-width: 1500px;}.st-key-am_symbol_box {height: 71px !important; min-height: 71px; position: relative; z-index: 30; overflow: visible;}
.st-key-am_symbol_box * {overflow: visible;}
.st-key-am_symbol_box iframe {position: absolute; top: 0; left: 0; width: 100%; z-index: 30;}
.am-header {display:flex; align-items:center; gap:.9rem;}
.am-header h1 {font-size:1.6rem; margin:0; padding:0; font-weight:700;}
.am-header p {margin:0; opacity:.7; font-size:.95rem;}
.am-clock {text-align:right; opacity:.8; font-size:.9rem; padding-top:.5rem;}
.am-tile {border-radius:12px; padding:.8rem .95rem; background:rgba(100,116,139,.08); height:100%;}
.am-tile.tone-up {background:rgba(22,163,74,.10);} .am-tile.tone-down {background:rgba(220,38,38,.09);}
.am-tile.tone-accent {background:rgba(37,99,235,.08);}
.am-tile .lbl {font-size:.8rem; opacity:.75; margin-bottom:.15rem;}
.am-tile .val {font-size:1.25rem; font-weight:700; line-height:1.3;}
.am-tile .sub {font-size:.8rem; opacity:.75; margin-top:.15rem;}
.am-bar {height:8px; border-radius:6px; background:rgba(100,116,139,.2); margin-top:.45rem; overflow:hidden;}
.am-bar > div {height:100%; border-radius:6px;}
.am-pill {display:inline-block; padding:.2rem .7rem; border-radius:999px; font-weight:600; font-size:.85rem;}
.am-price {font-size:1.9rem; font-weight:700;}
.am-change {font-weight:600; margin-left:.6rem;}
.am-muted {opacity:.7; font-size:.85rem;}
.am-rows {width:100%; border-collapse:collapse; font-size:.9rem;}
.am-rows td, .am-rows th {padding:.42rem .3rem; border-bottom:1px solid rgba(100,116,139,.18); text-align:left;}
.am-rows th {font-weight:600; opacity:.75; font-size:.8rem; background:rgba(100,116,139,.07);}
.am-glyph {display:inline-flex; width:1.6rem; height:1.6rem; border-radius:50%; align-items:center;
           justify-content:center; font-size:1rem; margin-right:.5rem; vertical-align:middle;}
.am-reasons {border-radius:12px; padding:.75rem 1rem; background:rgba(37,99,235,.07);}
.am-reasons ul {margin:.3rem 0 0 0; padding-left:1.1rem;} .am-reasons li {margin:.15rem 0; font-size:.9rem;}
.am-explain {display:flex; gap:.75rem; border-radius:12px; padding:.75rem .9rem; margin-bottom:.6rem;}
.am-explain .ico {font-size:1.25rem; width:2.1rem; height:2.1rem; border-radius:50%; display:flex;
                  align-items:center; justify-content:center; flex-shrink:0;}
.am-explain b {display:block; margin-bottom:.15rem;} .am-explain span {font-size:.87rem; opacity:.85;}
.am-legend {display:flex; gap:1.4rem; font-size:.8rem; opacity:.8; margin-top:-.3rem;}
.am-legend i {display:inline-block; width:1.4rem; height:.6rem; border-radius:3px; margin-right:.35rem;
              vertical-align:middle;}
.am-note {font-size:.78rem; opacity:.7; margin-top:.4rem;}
[data-testid="stSidebarHeader"] img {height: 3rem !important; max-width: 100%;}
section[data-testid="stSidebar"] .am-disclaimer {font-size:.82rem; border-radius:10px; padding:.75rem .85rem;
              background:rgba(255,255,255,.07); margin-top:1rem; line-height:1.45;}
</style>
"""


def inject_css():
    st.html(CSS)


PANEL_W = "20rem"


def side_panel_css(open_: bool):
    """Right-hand settings panel (container key `am_settings_panel`) that slides in below Streamlit's top
    bar, like the navigation sidebar on the left. When open, the page content makes room for it."""
    st.html(f"""<style>
.st-key-am_settings_panel {{position: fixed; top: 3.75rem; right: 0; bottom: 0; width: min({PANEL_W}, 100vw);
  z-index: 100; padding: 1rem 1.1rem 2rem; overflow-y: auto; background: var(--secondary-background-color, #e9eef7);
  border-left: 1px solid rgba(100,116,139,.25); box-shadow: -6px 0 18px rgba(15,23,42,.06);
  transition: transform .25s ease, visibility .25s;
  transform: translateX({"0" if open_ else "100%"}); visibility: {"visible" if open_ else "hidden"};}}
{f'[data-testid="stMainBlockContainer"] {{padding-right: calc({PANEL_W} + 1.5rem);}}' if open_ else ""}
@media (max-width: 900px) {{[data-testid="stMainBlockContainer"] {{padding-right: 1rem;}}}}
</style>""")


def tint(color: str, alpha: float) -> str:
    c = color.lstrip("#")
    return f"rgba({int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)},{alpha})"


def page_header(title: str, subtitle: str, right=None):
    """Logo + title + subtitle on the left; optional callable rendering extra widgets on the right."""
    left, mid, gear = st.columns([6, 2.2, 0.5], vertical_alignment="center")
    left.html(f'<div class="am-header"><div><h1>{escape(title)}</h1>'
              f'<p>{escape(subtitle)}</p></div></div>')
    if right:
        right(mid, gear)


def tile(label: str, value: str, sub: str = "", tone: str = "", help: str | None = None, bar: float | None = None,
         bar_tone: str = "up", where=st):
    """Small stat tile. `bar` (0-1) draws a progress bar under the value."""
    bar_html = (f'<div class="am-bar"><div style="width:{max(0., min(1., bar)) * 100:.0f}%;'
                f'background:{TONES[bar_tone]}"></div></div>') if bar is not None else ""
    title = f' title="{escape(help)}"' if help else ""
    where.html(f'<div class="am-tile tone-{tone}"{title}><div class="lbl">{escape(label)}'
               f'{" ⓘ" if help else ""}</div><div class="val">{escape(value)}</div>'
               f'{f"<div class=sub>{escape(sub)}</div>" if sub else ""}{bar_html}</div>')


def pill(text: str, tone: str = "neutral", where=st):
    where.html(f'<span class="am-pill" style="background:{tint(TONES[tone], .14)};color:{TONES[tone]}">'
               f'{escape(text)}</span>')


def price_header(price: float, change: float, change_pct: float, currency_symbol: str, sub: str, where=st):
    tone = UP if change >= 0 else DOWN
    arrow = "▲" if change >= 0 else "▼"
    where.html(f'<div><span class="am-price">{escape(currency_symbol)} {price:,.2f}</span>'
               f'<span class="am-change" style="color:{tone}">{change:+,.2f} ({change_pct:+.2%}) {arrow}</span></div>'
               f'<div class="am-muted">{escape(sub)}</div>')


def bullet_box(title: str, items: list[str], icon: str = "💡", where=st):
    lis = "".join(f"<li>{escape(i)}</li>" for i in items)
    where.html(f'<div class="am-reasons"><b>{icon} {escape(title)}</b><ul>{lis}</ul></div>')


def explain_card(title: str, body: str, icon: str, tone: str = "accent", where=st):
    c = TONES[tone]
    where.html(f'<div class="am-explain" style="background:{tint(c, .08)}"><div class="ico" '
               f'style="background:{tint(c, .18)};color:{c}">{icon}</div><div><b>{escape(title)}</b>'
               f'<span>{escape(body)}</span></div></div>')


def note(text: str, where=st):
    where.html(f'<div class="am-note">{escape(text)}</div>')


def legend(items: list[tuple[str, str]], where=st):
    where.html('<div class="am-legend">' + "".join(
        f'<span><i style="background:{c}"></i>{escape(t)}</span>' for t, c in items) + "</div>")


def deg_min(deg: float) -> str:
    d = int(deg)
    return f"{d}° {int(round((deg - d) * 60)) % 60:02d}'"


def planet_rows(positions: pd.DataFrame, bodies: list[str], extra: list[tuple[str, str, str]] = (), where=st):
    """Table of bodies with glyph, sign and degree-in-sign. `extra` = (glyph, label, value) rows."""
    rows = []
    for b in bodies:
        r = positions.loc[positions.body == b]
        if r.empty:
            continue
        r = r.iloc[0]
        g, c = PLANET_GLYPHS.get(b, ("•", NEUTRAL))
        retro = " ℞" if bool(r.get("retrograde", False)) else ""
        rows.append(f'<tr><td><span class="am-glyph" style="background:{tint(c, .18)};color:{c}">{g}</span>'
                    f'{escape(b.title())}</td><td>{escape(str(r["sign"]))}{retro}</td>'
                    f'<td style="text-align:right">{deg_min(float(r["degree_in_sign"]))}</td></tr>')
    for g, label, value in extra:
        rows.append(f'<tr><td><span class="am-glyph" style="background:{tint(NEUTRAL, .15)}">{g}</span>'
                    f'<b>{escape(label)}</b></td><td colspan="2">{escape(value)}</td></tr>')
    where.html(f'<table class="am-rows">{"".join(rows)}</table>')


def outcome_rows(df: pd.DataFrame, return_label: str, where=st):
    """Dates with a return and an Up/Down result (df columns: date, ret)."""
    body = "".join(
        f'<tr><td>{r.date:%b %d, %Y}</td><td style="color:{UP if r.ret > 0 else DOWN}">{r.ret:+.1%}</td>'
        f'<td style="color:{UP if r.ret > 0 else DOWN}">{"Up" if r.ret > 0 else "Down"}</td></tr>'
        for r in df.itertuples())
    where.html(f'<table class="am-rows"><tr><th>Date</th><th>{escape(return_label)}</th><th>Result</th></tr>'
               f'{body}</table>')


def candles_with_forecast(bars: pd.DataFrame, fc: pd.DataFrame | None = None, height: int = 330,
                          split_labels: tuple[str, str] = ("Actual", "Forecast")):
    """Candlestick chart (index = display-time timestamps) with an optional forecast path and band.
    `fc` columns: t, price, low, high (first row = the last actual close)."""
    d = bars[["open", "high", "low", "close"]].rename_axis("t").reset_index().dropna()
    step = d.t.diff().median() if len(d) > 1 else pd.Timedelta(days=1)
    half = step * 0.35
    d = d.assign(t0=d.t - half, t1=d.t + half)
    color = alt.condition("datum.open <= datum.close", alt.value(UP), alt.value(DOWN))
    tips = [alt.Tooltip("t:T", title="Time"), alt.Tooltip("open:Q", format=",.2f"),
            alt.Tooltip("high:Q", format=",.2f"), alt.Tooltip("low:Q", format=",.2f"),
            alt.Tooltip("close:Q", format=",.2f")]
    lo = float(d.low.min()) if fc is None else float(min(d.low.min(), fc.low.min()))
    hi = float(d.high.max()) if fc is None else float(max(d.high.max(), fc.high.max()))
    pad = (hi - lo) * 0.05 or hi * 0.01
    y = alt.Y("low:Q", scale=alt.Scale(domain=[lo - pad, hi + pad], zero=False), title=None,
              axis=alt.Axis(format=",.0f"))
    base = alt.Chart(d).encode(tooltip=tips)
    layers = [base.mark_rule().encode(x=alt.X("t:T", title=None), y=y, y2="high:Q", color=color),
              base.mark_rect().encode(x="t0:T", x2="t1:T", y="open:Q", y2="close:Q", color=color)]
    if fc is not None and len(fc):
        ftips = [alt.Tooltip("t:T", title="Date"), alt.Tooltip("price:Q", format=",.2f", title="expected"),
                 alt.Tooltip("low:Q", format=",.2f", title="low 80%"), alt.Tooltip("high:Q", format=",.2f",
                                                                                  title="high 80%")]
        f = alt.Chart(fc)
        layers = [f.mark_area(opacity=.18, color=ACCENT).encode(x="t:T", y="low:Q", y2="high:Q", tooltip=ftips),
                  *layers,
                  f.mark_line(strokeDash=[5, 4], color=ACCENT, point=alt.OverlayMarkDef(size=18, color=ACCENT))
                  .encode(x="t:T", y="price:Q", tooltip=ftips)]
        split = pd.DataFrame({"t": [fc.t.iloc[0]]})
        layers.append(alt.Chart(split).mark_rule(strokeDash=[3, 3], color=NEUTRAL).encode(x="t:T"))
        lab = pd.DataFrame({"t": [d.t.iloc[len(d) // 2], fc.t.iloc[len(fc) // 2]], "y": [lo - pad * .5] * 2,
                            "txt": list(split_labels), "c": [NEUTRAL, ACCENT]})
        layers.append(alt.Chart(lab).mark_text(fontSize=12, dy=-4).encode(
            x="t:T", y="y:Q", text="txt:N", color=alt.Color("c:N", scale=None)))
    return alt.layer(*layers).properties(height=height)


def growth_path(base: float, exp_ret: float, lo_ret: float, hi_ret: float, times: list) -> pd.DataFrame:
    """Spread a window's log return evenly over its sessions (display only): rows t, price, low, high."""
    n = len(times) - 1
    f = np.arange(len(times)) / max(n, 1)
    return pd.DataFrame({"t": times, "price": base * np.exp(exp_ret * f),
                         "low": base * np.exp(lo_ret * f), "high": base * np.exp(hi_ret * f)})
