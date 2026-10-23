"""One feature group per catalog category, for the dashboard's forecast engine (Model Comparison,
Backtesting, Future Forecast). Column names are the catalog's `Feature Name`s.

Row t = decision after the close of session t; the target window starts at session t+gap+1.
Availability per group (all point-in-time):
  c01 c02 c03 c04   the instrument's own prices/volume up to close(t)
  c05 c09           benchmark index and India VIX up to close(t) (same exchange, same session)
  c10 c11           overseas markets: value dated strictly before t (they close after Asian markets);
                    c11's currency pairs are exchange-specific (see REGIONAL_FX below)
  c12 c14           GPR / NASA POWER, re-dated to publication (src/risk_external.py); c14 (weather)
                    is built from India-only stations and is only offered for NSE/BSE listings
  c16 c17 c18       planetary positions at the first target session (deterministic, known in advance)
  c19               calendar of the first target session (exchange calendars are published in advance)
  c21               products of the groups above (same availability as their parents)
Categories without a point-in-time data source are reported as unavailable, never approximated:
  00, 22  metadata / data quality: shown as a coverage table, not fed to models
  06, 07, 08, 13, 15, 20  need licensed or archived point-in-time feeds (not available here)
"""
import numpy as np
import pandas as pd

ANN = np.sqrt(252)

CATEGORIES = {
    "c01": "01 Price, returns & OHLCV", "c02": "02 Technical indicators & trend",
    "c03": "03 Volume, liquidity & microstructure", "c04": "04 Volatility, drawdown & risk",
    "c05": "05 Market, sector & factor context", "c09": "09 Derivatives & positioning",
    "c10": "10 Macroeconomics, rates & policy", "c11": "11 Currencies, commodities, bonds & credit",
    "geopolitics": "12 Geopolitics, conflict & country risk",
    "weather": "14 Weather, climate & natural hazards (India only)",
    "c16": "16 Astrology - astronomical positions", "c17": "17 Astrology - aspects & Western astrology",
    "c18": "18 Astrology - Vedic calendar & techniques", "c19": "19 Calendar, seasonality & event timing",
    "c21": "21 Cross-domain interactions & regimes",
}
UNAVAILABLE = {
    "00 Data keys & point-in-time metadata": "Identifiers and timestamps, not predictors (catalog: do not auto-feed). Shown in the data coverage table.",
    "06 Company fundamentals & valuation": "Needs point-in-time (as-reported) statements; Yahoo only has today's restated values, which would leak.",
    "07 Corporate events & earnings calendar": "Needs an archived event calendar with announcement timestamps.",
    "08 Investor flows, ownership & analyst revisions": "Needs FII/DII flow history and point-in-time analyst estimates (licensed).",
    "13 News, sentiment & alternative information": "Needs a timestamped news archive (licensed feed or GDELT ingestion).",
    "15 Geography, trade & supply-chain exposure": "Needs company/country exposure data with release dates.",
    "20 Alternative data & sector-specific signals": "Needs industry activity series with release dates.",
    "22 Data quality, lineage & monitoring": "Monitoring fields, not predictors. Shown in the data coverage table.",
}
INDIA = {"NSE", "BSE"}
BENCHMARK = {"NSE": "^NSEI", "BSE": "^BSESN", "NYSE": "^GSPC", "NASDAQ": "^GSPC", "LSE": "^FTSE"}
# Overseas markets used in categories 10 and 11 (Yahoo symbols); all lagged one session.
# Commodities, US yields and US equities are global risk factors relevant to any listing, so they're
# fetched for every exchange. Currency pairs are only meaningful relative to the listing's own
# currency, so they're chosen per exchange below instead of a single fixed list for everyone.
GLOBAL_SERIES_BASE = {"usd_jpy": "JPY=X", "dxy": "DX-Y.NYB", "brent": "BZ=F", "wti": "CL=F",
                      "natgas": "NG=F", "gold": "GC=F", "silver": "SI=F", "copper": "HG=F",
                      "us_10y": "^TNX", "us_3m": "^IRX", "us_equity": "^GSPC"}
REGIONAL_FX = {
    "NSE": {"usd_inr": "INR=X", "eur_inr": "EURINR=X"}, "BSE": {"usd_inr": "INR=X", "eur_inr": "EURINR=X"},
    "NYSE": {"eur_usd": "EURUSD=X", "gbp_usd": "GBPUSD=X"}, "NASDAQ": {"eur_usd": "EURUSD=X", "gbp_usd": "GBPUSD=X"},
    "LSE": {"gbp_usd": "GBPUSD=X", "eur_gbp": "EURGBP=X"},
}


# ---------------------------------------------------------------- own prices (01-04)
def _adjusted_ohlc(px: pd.DataFrame) -> pd.DataFrame:
    """Scale raw open/high/low by price/close so split and dividend days do not show up as gaps."""
    k = (px["price"] / px["close"]).where(px["close"] > 0)
    return pd.DataFrame({"open": px["open"] * k, "high": px["high"] * k, "low": px["low"] * k,
                         "close": px["price"]}, index=px.index).astype(float)


def price_returns(px):            # 01
    a = _adjusted_ohlc(px)
    o, h, l, c = a["open"], a["high"], a["low"], a["close"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=px.index)
    f["log_return_1d"] = r
    for k in (2, 3, 5, 10, 20, 60, 120, 252):
        f[f"return_{k}d"] = r.rolling(k).sum()
    f["overnight_return_1d"] = np.log(o / c.shift(1))
    f["intraday_return_1d"] = np.log(c / o)
    f["ohlc_range_pct"] = (h - l) / c
    rng = (h - l).replace(0, np.nan)
    f["close_location_value"] = (((c - l) - (h - c)) / rng).where(rng.notna() | c.isna(), 0.0)   # no range: neutral
    f["upper_shadow_pct"] = (h - np.maximum(o, c)) / c
    f["lower_shadow_pct"] = (np.minimum(o, c) - l) / c
    for k in (20, 60):
        f[f"up_day_fraction_{k}d"] = (r > 0).astype(float).rolling(k).mean()
    f["best_return_20d"], f["worst_return_20d"] = r.rolling(20).max(), r.rolling(20).min()
    f["return_skew_60d"], f["return_kurtosis_60d"] = r.rolling(60).skew(), r.rolling(60).kurt()
    f["return_autocorrelation_20d"] = r.rolling(20).corr(r.shift(1))
    gap = f["overnight_return_1d"]
    f["open_gap_zscore_60d"] = (gap - gap.rolling(60).mean()) / gap.rolling(60).std()
    for k in (20, 252):
        f[f"price_vs_sma_{k}d"] = c / c.rolling(k).mean() - 1
    return f


def technical(px):                # 02
    a = _adjusted_ohlc(px)
    h, l, c = a["high"], a["low"], a["close"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=px.index)
    for k in (2, 14):
        gain, loss = r.clip(lower=0).rolling(k).mean(), (-r.clip(upper=0)).rolling(k).mean()
        # = 100 - 100/(1+RS); no losses -> 100, no movement at all -> 50
        f[f"rsi_{k}d"] = (100 * gain / (gain + loss)).where(gain + loss > 0, 50.0).where(gain.notna())
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / c                                       # scaled by price: comparable over time
    sig = macd.ewm(span=9, adjust=False).mean()
    f["macd_line"], f["macd_signal"], f["macd_histogram"] = macd, sig, macd - sig
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bollinger_band_width_20d"] = 4 * s20 / m20
    # bollinger_percent_b_20d stands in for 3 near-duplicate "distance from mean" indicators:
    # it correlates 1.000 with a rolling price z-score and 0.99 with the commodity channel index,
    # so williams_r_14d/cci_20d/rolling_price_zscore_20d are dropped rather than kept alongside it.
    f["bollinger_percent_b_20d"] = (c - (m20 - 2 * s20)) / (4 * s20)
    hh, ll = h.rolling(14).max(), l.rolling(14).min()
    f["stochastic_k_14d"] = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    f["atr_14d_pct"] = atr / c
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c.index).rolling(14).sum()
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index).rolling(14).sum()
    tr14 = tr.rolling(14).sum()
    pdi, mdi = 100 * pdm / tr14, 100 * mdm / tr14
    # no directional movement at all (pdi+mdi==0) -> neutral "no trend" (0.0), not NaN, same pattern
    # as the RSI/close_location_value fixes elsewhere in this file.
    f["adx_14d"] = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).fillna(0.0).rolling(14).mean()
    f["roc_20d"] = c / c.shift(20) - 1
    f["distance_from_52w_high"] = c / h.rolling(252).max() - 1
    f["distance_from_52w_low"] = c / l.rolling(252).min() - 1
    f["sma_50_200_spread"] = c.rolling(50).mean() / c.rolling(200).mean() - 1
    x = np.arange(20) - 9.5
    f["linear_trend_slope_20d"] = np.log(c).rolling(20).apply(lambda y: np.dot(x, y - y.mean()) / np.dot(x, x), raw=True)
    return f


def volume_liquidity(px):         # 03
    a = _adjusted_ohlc(px)
    c, v = a["close"], px["volume"].astype(float)
    r = np.log(c).diff()
    f = pd.DataFrame(index=px.index)
    lv = np.log1p(v)
    f["volume_change_5d"] = lv - lv.shift(5)
    for k in (20, 60):
        f[f"volume_ratio_{k}d"] = np.log((v + 1) / (v.rolling(k).mean() + 1))
    obv = (np.sign(r).fillna(0) * v).cumsum()
    # zero 20-day volume (e.g. a stray no-trade day) -> neutral "no net flow" (0.0), not NaN.
    f["obv_change_20d"] = ((obv - obv.shift(20)) / v.rolling(20).sum().replace(0, np.nan)).fillna(0.0)
    tp = (a["high"] + a["low"] + c) / 3
    mf = tp * v
    pos = mf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    # no negative money flow at all -> neutral midpoint (50), same convention as rsi_Xd above,
    # instead of leaving it undefined.
    f["money_flow_index_14d"] = (100 - 100 / (1 + pos / neg.replace(0, np.nan))).where((pos + neg) > 0, 50.0)
    clv = (((c - a["low"]) - (a["high"] - c)) / (a["high"] - a["low"]).replace(0, np.nan)).fillna(0.0)
    f["chaikin_money_flow_20d"] = (clv * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    # zero price*volume on a stray day -> neutral 0.0 contribution instead of poisoning the whole
    # trailing 20-day mean (rolling() with the default min_periods needs every value non-NaN).
    f["amihud_illiquidity_20d"] = np.log1p(1e9 * (r.abs() / (c * v).replace(0, np.nan)).fillna(0.0).rolling(20).mean())
    f["zero_return_day_fraction_20d"] = (r == 0).astype(float).rolling(20).mean()
    return f


def volatility_risk(px):          # 04
    a = _adjusted_ohlc(px)
    o, h, l, c = a["open"], a["high"], a["low"], a["close"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=px.index)
    for k in (5, 20, 60, 252):
        f[f"realized_volatility_{k}d"] = np.log(r.rolling(k).std() * ANN)
    f["ewma_volatility"] = np.log(np.sqrt((r ** 2).ewm(alpha=0.06).mean()) * ANN)
    f["downside_volatility_20d"] = np.log(np.sqrt((r.clip(upper=0) ** 2).rolling(20).mean()) * ANN + 1e-6)
    f["upside_volatility_20d"] = np.log(np.sqrt((r.clip(lower=0) ** 2).rolling(20).mean()) * ANN + 1e-6)
    for k in (20, 60):
        f[f"max_drawdown_{k}d"] = (c / c.rolling(k).max() - 1).rolling(k).min()
    f["drawdown_from_rolling_peak"] = c / c.rolling(252).max() - 1
    f["historical_var_95_20d"] = r.rolling(20).quantile(0.05)
    f["historical_cvar_95_60d"] = r.rolling(60).apply(lambda x: x[x <= np.quantile(x, 0.05)].mean(), raw=True)
    hl, co = np.log(h / l), np.log(c / o)
    # parkinson_volatility_20d dropped: corr 0.996 with garman_klass_volatility_20d (the latter is
    # the more complete OHLC estimator - also uses the open/close range, not just high/low).
    f["garman_klass_volatility_20d"] = np.log(np.sqrt((0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2)
                                                     .clip(lower=0).rolling(20).mean()) * ANN + 1e-6)
    rv20 = r.rolling(20).std()
    f["volatility_of_volatility_60d"] = np.log(rv20).rolling(60).std()
    lrv = np.log(rv20)
    f["volatility_regime_zscore"] = (lrv - lrv.rolling(252).mean()) / lrv.rolling(252).std()
    z = r / r.rolling(60).std().shift(1)
    f["jump_frequency_60d"] = (z.abs() > 3).astype(float).rolling(60).mean()
    f["tail_loss_count_60d"] = (z < -2).astype(float).rolling(60).sum()
    return f


# ---------------------------------------------------------------- market context, derivatives (05, 09)
def market_context(px, bench: pd.Series | None, vix: pd.Series | None):
    f = pd.DataFrame(index=px.index)
    r = np.log(px["price"]).diff()
    if bench is not None:
        b = np.log(bench.reindex(px.index)).diff()
        for k in (1, 5, 20, 60):
            f[f"benchmark_return_{k}d"] = b.rolling(k).sum()
        f["benchmark_realized_vol_20d"] = np.log(b.rolling(20).std() * ANN)
        cov = r.rolling(60).cov(b)
        f["beta_vs_market_60d"] = cov / b.rolling(60).var()
        f["correlation_to_market_60d"] = r.rolling(60).corr(b)
        f["stock_minus_market_return_20d"] = r.rolling(20).sum() - b.rolling(20).sum()
    if vix is not None:
        v = vix.reindex(px.index)
        f["india_vix_level"] = np.log(v)
        f["india_vix_change_1d"] = np.log(v).diff()
        f["india_vix_change_5d"] = np.log(v).diff(5)
    return f


def derivatives(px, vix: pd.Series | None):
    f = pd.DataFrame(index=px.index)
    if vix is not None:
        r = np.log(px["price"]).diff()
        f["realized_minus_implied_vol"] = r.rolling(20).std() * ANN * 100 - vix.reindex(px.index)
    return f


# ---------------------------------------------------------------- overseas markets (10, 11)
def rates_macro(g: dict):
    f = pd.DataFrame(index=next(iter(g.values())).index)
    if "us_10y" in g:
        y10 = g["us_10y"]
        f["us_10y_treasury_yield"] = y10
        f["us_10y_yield_change_20d"] = y10 - y10.shift(20)
        if "us_3m" in g:
            f["us_3m_tbill_yield"] = g["us_3m"]
            f["us_yield_curve_slope_10y_3m"] = y10 - g["us_3m"]
    return f


def currencies_commodities(g: dict):
    f = pd.DataFrame(index=next(iter(g.values())).index)
    lr = {k: np.log(v.where(v > 0)).diff() for k, v in g.items()}
    if "usd_inr" in lr:
        f["usd_inr_return_1d"], f["usd_inr_return_5d"] = lr["usd_inr"], lr["usd_inr"].rolling(5).sum()
        f["usd_inr_volatility_20d"] = np.log(lr["usd_inr"].rolling(20).std() * ANN)
    if "brent" in lr:
        f["brent_return_1d"], f["brent_return_5d"] = lr["brent"], lr["brent"].rolling(5).sum()
        f["brent_volatility_20d"] = np.log(lr["brent"].rolling(20).std() * ANN)
    for k, name in [("eur_inr", "eur_inr_return_5d"), ("eur_usd", "eur_usd_return_5d"),
                    ("gbp_usd", "gbp_usd_return_5d"), ("eur_gbp", "eur_gbp_return_5d"),
                    ("dxy", "dxy_return_5d"), ("wti", "wti_return_5d"),
                    ("natgas", "natural_gas_return_5d"), ("gold", "gold_return_5d"), ("silver", "silver_return_5d"),
                    ("copper", "copper_return_5d")]:
        if k in lr:
            f[name] = lr[k].rolling(5).sum()
    if "usd_jpy" in lr:
        f["jpy_usd_return_5d"] = -lr["usd_jpy"].rolling(5).sum()      # JPY per USD inverted
    if "us_equity" in lr:
        f["us_equity_return_1d_lagged"] = lr["us_equity"]
        f["us_equity_volatility_20d_lagged"] = np.log(lr["us_equity"].rolling(20).std() * ANN)
    return f


# ---------------------------------------------------------------- astrology (16, 17, 18)
PAIR_BODIES = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn"]
ASPECT_ANGLES = {"conjunction": 0, "sextile": 60, "square": 90, "trine": 120, "opposition": 180}
# Outer planets move so little in any one instrument's history that their longitude is nearly a
# straight line over the whole sample and nearly collinear WITH EACH OTHER (confirmed empirically:
# neptune/uranus/pluto sin/cos pairs correlate 0.94-0.99 across bodies, not just within one) -
# they add no separable signal and are dropped entirely rather than reduced.
REDUNDANT_SLOW_BODIES = {"uranus", "neptune", "pluto"}
# Jupiter/Saturn/the lunar nodes barely move day-to-day (<1% of their own range) at 1-20 session
# horizons - their longitude is effectively frozen, but a retrograde FLIP is a real, fast event,
# so the flag stays while the near-constant longitude/speed does not.
NEAR_STATIC_LONGITUDE_BODIES = {"jupiter", "saturn", "rahu"}


def astro_positions(sky: pd.DataFrame):
    """`sky`: one row per decision row, describing the first target session."""
    f = pd.DataFrame(index=sky.index)
    bodies = [c[:-4] for c in sky.columns if c.endswith("_lon") and not c.startswith("ketu")]
    for b in bodies:
        if b in REDUNDANT_SLOW_BODIES:
            continue
        rad = np.deg2rad(sky[f"{b}_lon"].astype(float))
        if b == "rahu":            # catalog: rahu_ketu_axis_longitude (Ketu is always opposite)
            continue                # near-static (see NEAR_STATIC_LONGITUDE_BODIES); no retro flag to keep either
        name = b
        if b not in NEAR_STATIC_LONGITUDE_BODIES:
            f[f"{name}_longitude_sin"], f[f"{name}_longitude_cos"] = np.sin(rad), np.cos(rad)
            # sun's speed duplicates its own sin (0.98 correlated: Earth's orbit is nearly
            # circular, so speed phase and position phase move together over any one sample).
            if f"{b}_speed" in sky and b != "sun":
                f[f"{name}_longitude_speed_deg_day"] = sky[f"{b}_speed"].astype(float)
        if f"{b}_retro" in sky:
            f[f"{name}_retrograde_flag"] = sky[f"{b}_retro"].astype(float)
    e = np.deg2rad(sky["elongation"].astype(float))
    f["moon_phase_sin"], f["moon_phase_cos"] = np.sin(e), np.cos(e)
    # moon_illumination_fraction is a 1:1 function of moon_phase_cos (corr 1.000) - redundant.
    retro = [c for c in sky.columns if c.endswith("_retro")]
    f["planetary_retrograde_count"] = sky[retro].sum(axis=1).astype(float)
    return f


def astro_aspects(sky: pd.DataFrame):
    """Only the raw separation per pair: the 5 named "orb" distances the catalog also lists
    (conjunction/sextile/square/trine/opposition) are each a fixed shift of this same number and
    correlate ~1.000 with it and each other in practice - tree models recover any aspect-closeness
    pattern from the raw angle alone, so keeping all 6 per pair was pure redundancy. The sun-moon
    pair is skipped entirely: it's the same Sun-Moon angle astro_positions already encodes as
    moon_phase_sin/cos (corr 0.99 with this pair's separation)."""
    f = {}
    for i, a in enumerate(PAIR_BODIES):
        for b in PAIR_BODIES[i + 1:]:
            if {a, b} == {"sun", "moon"}:
                continue
            d = np.abs(sky[f"{a}_lon"].astype(float) - sky[f"{b}_lon"].astype(float)) % 360
            f[f"{a}_{b}_angular_separation_deg"] = np.minimum(d, 360 - d)
    return pd.DataFrame(f, index=sky.index)


def astro_vedic(sky: pd.DataFrame):
    """Panchang elements from the zodiac chosen in settings (sidereal for Vedic use)."""
    f = pd.DataFrame(index=sky.index)
    sun, moon = sky["sun_lon"].astype(float), sky["moon_lon"].astype(float)
    elong = sky["elongation"].astype(float)
    # moon_rashi_id dropped: corr 0.996 with moon_nakshatra_id (same lunar longitude, coarser bucket).
    f["sun_rashi_id"] = (sun // 30).astype(float)
    nak = moon / (360 / 27)
    f["moon_nakshatra_id"] = np.floor(nak)
    f["moon_nakshatra_pada"] = np.floor((nak % 1) * 4) + 1
    f["tithi_id"] = np.floor(elong / 12) + 1
    f["yoga_id"] = np.floor(((sun + moon) % 360) / (360 / 27)) + 1
    k = np.floor(elong / 6).astype(int)                      # 60 half-tithis -> 11 karanas
    f["karana_id"] = np.select([k == 0, k >= 57], [10, k - 50], (k - 1) % 7).astype(float)
    f["paksha_flag"] = (elong >= 180).astype(float)          # 1 = Krishna (waning)
    f["new_moon_session_flag"] = ((elong < 12) | (elong >= 348)).astype(float)
    f["full_moon_session_flag"] = ((elong >= 168) & (elong < 192)).astype(float)
    return f


# ---------------------------------------------------------------- calendar (19), interactions (21)
def calendar(target_sessions: pd.Series, all_sessions: pd.DatetimeIndex, exchange_code: str):
    """Calendar of the first target session; month/quarter-end distances in trading sessions."""
    d = pd.DatetimeIndex(target_sessions.to_numpy())
    f = pd.DataFrame(index=target_sessions.index)
    f["day_of_week_sin"], f["day_of_week_cos"] = np.sin(2 * np.pi * d.dayofweek / 5), np.cos(2 * np.pi * d.dayofweek / 5)
    f["month_sin"], f["month_cos"] = np.sin(2 * np.pi * d.month / 12), np.cos(2 * np.pi * d.month / 12)
    pos = all_sessions.get_indexer(d)
    s = pd.Series(all_sessions)
    month_last = s.groupby([s.dt.year, s.dt.month]).transform("max")
    q_last = s.groupby([s.dt.year, s.dt.quarter]).transform("max")
    idx_of = pd.Series(np.arange(len(s)), index=all_sessions)
    f["month_end_days"] = np.where(pos >= 0, idx_of.reindex(month_last.iloc[pos.clip(0)].to_numpy()).to_numpy() - pos, np.nan)
    f["quarter_end_days"] = np.where(pos >= 0, idx_of.reindex(q_last.iloc[pos.clip(0)].to_numpy()).to_numpy() - pos, np.nan)
    fy_month = 3 if exchange_code in INDIA else 12
    f["financial_year_end_flag"] = (d.month == fy_month).astype(float)
    gaps_after = pd.Series(np.r_[np.diff(all_sessions).astype("timedelta64[D]").astype(int), 1], index=all_sessions)
    gaps_before = pd.Series(np.r_[1, np.diff(all_sessions).astype("timedelta64[D]").astype(int)], index=all_sessions)
    # a weekday closure next to the session (a weekend alone is a 3-day gap after Friday)
    f["pre_holiday_session_flag"] = ((gaps_after.reindex(d).to_numpy() > 1) & ~((d.dayofweek == 4) & (gaps_after.reindex(d).to_numpy() == 3))).astype(float)
    f["first_session_after_holiday"] = ((gaps_before.reindex(d).to_numpy() > 1) & ~((d.dayofweek == 0) & (gaps_before.reindex(d).to_numpy() == 3))).astype(float)
    return f


def interactions(groups: dict):
    f = pd.DataFrame(index=next(iter(groups.values())).index)
    c01, c04 = groups.get("c01"), groups.get("c04")
    if c01 is not None and c04 is not None:
        f["stock_momentum_x_market_volatility"] = c01["return_20d"] * c04["volatility_regime_zscore"]
    if "c09" in groups and "c05" in groups and "india_vix_level" in groups["c05"]:
        f["iv_x_realized_vol_gap"] = groups["c05"]["india_vix_level"] * groups["c09"]["realized_minus_implied_vol"] / 100
    if "c16" in groups and c04 is not None:
        f["moon_phase_x_market_volatility"] = groups["c16"]["moon_phase_cos"] * c04["volatility_regime_zscore"]
    return f


# ---------------------------------------------------------------- assembly
def _load_close(symbol: str, end: str) -> pd.Series:
    from .risk_data import load_series
    return load_series(symbol, "2000-01-01", end, reuse_newer=True)["close"]


def build_category_groups(px: pd.DataFrame, sky_target: pd.DataFrame, target_sessions: pd.Series,
                          all_sessions: pd.DatetimeIndex, exchange_code: str, symbol: str,
                          fetch=True) -> tuple[dict, list[str]]:
    """{group key: frame aligned to px.index} + notes. `fetch=False` (synthetic prices) skips every
    download-based group. Every failure drops just that group, with a note."""
    from .risk_data import align_to_sessions
    groups, notes = {}, []
    groups["c01"], groups["c02"], groups["c04"] = price_returns(px), technical(px), volatility_risk(px)
    if (px["volume"].tail(500) > 0).mean() > 0.9:
        groups["c03"] = volume_liquidity(px)
    else:
        notes.append("03 Volume: this instrument's feed reports no volume on many days (typical for indices); group skipped.")
    end = str(px.index[-1].date())
    india = exchange_code in INDIA
    if fetch:
        try:
            bsym = BENCHMARK.get(exchange_code)
            bench = None if bsym in (None, symbol) else align_to_sessions(_load_close(bsym, end), px.index, "same_session")
            vix = align_to_sessions(_load_close("^INDIAVIX", end), px.index, "same_session") if india else None
            c05 = market_context(px, bench, vix)
            if c05.shape[1]:
                groups["c05"] = c05
            if vix is not None:
                groups["c09"] = derivatives(px, vix)
        except Exception as e:
            notes.append(f"05/09 benchmark or VIX data unavailable ({type(e).__name__}).")
        try:
            g = {}
            series = {**GLOBAL_SERIES_BASE, **REGIONAL_FX.get(exchange_code, {})}
            for k, sym in series.items():
                try:
                    g[k] = align_to_sessions(_load_close(sym, end), px.index, "next_session")
                except Exception:
                    notes.append(f"{sym} unavailable; its features are skipped.")
            if g:
                groups["c10"], groups["c11"] = rates_macro(g), currencies_commodities(g)
                if not groups["c10"].shape[1]:
                    groups.pop("c10")
        except Exception as e:
            notes.append(f"10/11 overseas market data unavailable ({type(e).__name__}).")
    else:
        notes.append("Downloaded categories (05, 09-12, 14) are not run on synthetic prices.")
    if not india:
        notes.append("09 Derivatives uses India VIX and is only built for NSE/BSE listings.")
    groups["c16"], groups["c17"], groups["c18"] = (astro_positions(sky_target), astro_aspects(sky_target),
                                                   astro_vedic(sky_target))
    groups["c19"] = calendar(target_sessions, all_sessions, exchange_code)
    c21 = interactions(groups)
    if c21.shape[1]:
        groups["c21"] = c21
    groups = {k: v.replace([np.inf, -np.inf], np.nan).astype(float) for k, v in groups.items()}
    return groups, notes


def coverage_table(groups: dict) -> pd.DataFrame:
    """Categories 00/22 in practice: lineage and data quality for every group (not fed to models)."""
    rows = []
    for k, f in groups.items():
        ok = f.notna().all(axis=1)
        recent = f.tail(756)
        rows.append({"category": CATEGORIES.get(k, k), "features": f.shape[1],
                     "first_complete_session": str(ok.idxmax().date()) if ok.any() else None,
                     "missing_last_3y_pct": round(100 * float(recent.isna().any(axis=1).mean()), 2)})
    for cat, why in UNAVAILABLE.items():
        rows.append({"category": cat, "features": 0, "first_complete_session": None,
                     "missing_last_3y_pct": None, "note": why})
    return pd.DataFrame(rows).sort_values("category").reset_index(drop=True)
