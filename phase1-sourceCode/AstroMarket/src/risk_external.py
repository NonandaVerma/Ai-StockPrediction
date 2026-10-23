"""Round-2 external data: geopolitical risk (GPR) and weather (NASA POWER), point-in-time.

Availability rules (pre-registered for round 2, before any round-2 result was seen):
  GPR daily (Caldara & Iacoviello)   value for day d usable from d + 7 calendar days
  GPR monthly, India (GPRC_IND)      month m usable from end of m + 15 calendar days
  NASA POWER daily weather           value for day d usable from d + 5 calendar days
Weather anomalies are measured against a climatology built from PRIOR YEARS ONLY (an expanding
same-day-of-year mean), never from the full sample.

Known limitation: we only hold today's vintage of both sources. GPR is revised as newspaper
archives change, and POWER history is reprocessed reanalysis, so past values are close to but not
exactly what was published at the time.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .risk_data import align_to_sessions

EXT_DIR = Path("data/raw/external")
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/{}"
GPR_DAILY, GPR_MONTHLY = "data_gpr_daily_recent.dta", "data_gpr_export.dta"
GPR_DAILY_LAG_DAYS, GPR_MONTHLY_LAG_DAYS, WEATHER_LAG_DAYS = 7, 15, 5

# 10 locations spread over India's main population / agricultural regions, equal weight.
WEATHER_POINTS = {
    "delhi": (28.61, 77.21), "mumbai": (19.08, 72.88), "kolkata": (22.57, 88.36), "chennai": (13.08, 80.27),
    "nagpur": (21.15, 79.09), "lucknow": (26.85, 80.95), "bhopal": (23.26, 77.41), "ahmedabad": (23.02, 72.57),
    "hyderabad": (17.39, 78.49), "patna": (25.59, 85.14),
}
POWER_URL = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR,T2M"
             "&community=AG&longitude={lon}&latitude={lat}&start={start}&end={end}&format=CSV")
WEATHER_START = "19900101"


def _download(url: str, timeout: int = 180) -> bytes:
    import requests     # verifies TLS certificates (curl here fails behind the corporate proxy)
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def _latest_cached(pattern: str) -> Path | None:
    files = sorted(EXT_DIR.glob(pattern))
    return files[-1] if files else None


def load_gpr(live: bool = False, max_age_days: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """live=False: the pinned files used by the research pipeline (reproducible).
    live=True: dated copies for the dashboard (`<name>_<YYYYMMDD>.dta`), refreshed when older than
    `max_age_days`; on a download failure the newest cached copy is used."""
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    today = pd.Timestamp.today().normalize()
    for f in (GPR_DAILY, GPR_MONTHLY):
        if not live:
            p = EXT_DIR / f
            if not p.exists():
                p.write_bytes(_download(GPR_URL.format(f)))
        else:
            stem = f.removesuffix(".dta")
            p = _latest_cached(f"{stem}_*.dta")
            stale = p is None or (today - pd.Timestamp(p.stem.rsplit("_", 1)[-1])).days > max_age_days
            if stale:
                try:
                    new = EXT_DIR / f"{stem}_{today:%Y%m%d}.dta"
                    new.write_bytes(_download(GPR_URL.format(f)))
                    p = new
                except Exception:
                    p = p or (EXT_DIR / f if (EXT_DIR / f).exists() else None)
                    if p is None:
                        raise
        out.append(pd.read_stata(p))
    daily, monthly = out
    daily = daily.set_index(pd.to_datetime(daily["date"]))[["GPRD", "GPRD_ACT", "GPRD_THREAT"]].astype(float)
    monthly = monthly.set_index(pd.to_datetime(monthly["month"]))[["GPR", "GPRC_IND"]].astype(float)
    return daily.sort_index(), monthly.sort_index()


def load_weather(end: str, live: bool = False) -> pd.DataFrame:
    """Daily all-India (equal-weight mean of WEATHER_POINTS) precipitation (mm/day) and 2 m temperature (C).
    live=True (dashboard): a cached file that runs past `end` is reused instead of downloading again."""
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    stop = pd.Timestamp(end).strftime("%Y%m%d")
    frames = []
    for name, (lat, lon) in WEATHER_POINTS.items():
        p = EXT_DIR / f"power_{name}_{WEATHER_START}_{stop}.csv"
        newer = [] if p.exists() or not live else             sorted(q for q in EXT_DIR.glob(f"power_{name}_{WEATHER_START}_*.csv") if q.stem.rsplit("_", 1)[-1] > stop)
        if newer:
            p = newer[0]
        elif not p.exists():
            try:
                raw = _download(POWER_URL.format(lat=lat, lon=lon, start=WEATHER_START, end=stop)).decode()
                p.write_text(raw.split("-END HEADER-", 1)[-1].strip(), encoding="utf-8")
            except Exception:
                p = _latest_cached(f"power_{name}_{WEATHER_START}_*.csv")   # older but usable; stale values expire
                if p is None:
                    raise
        df = pd.read_csv(p)
        df.index = pd.to_datetime(df["YEAR"].astype(str) + df["DOY"].astype(str).str.zfill(3), format="%Y%j")
        frames.append(df.loc[:pd.Timestamp(end), ["PRECTOTCORR", "T2M"]].replace(-999.0, np.nan))
    w = pd.concat(frames).groupby(level=0).mean()
    w.columns = ["precip_mm", "t2m_c"]
    return w


def prior_years_climatology(x: pd.Series, window: int) -> pd.Series:
    """For each date, the mean of `window`-day rolling values on the same day-of-year in strictly
    earlier years (expanding). NaN until 5 prior years exist."""
    roll = x.rolling(window, min_periods=window).mean()
    df = pd.DataFrame({"v": roll, "doy": np.minimum(roll.index.dayofyear, 365), "yr": roll.index.year})
    piv = df.pivot_table(index="yr", columns="doy", values="v")
    prior = piv.expanding(min_periods=5).mean().shift(1)          # years < current year only
    clim = prior.stack().rename("clim").reset_index()
    out = df.reset_index().merge(clim, on=["yr", "doy"], how="left").set_index("index")["clim"]
    return out.reindex(x.index)


def weather_features_daily(w: pd.DataFrame) -> pd.DataFrame:
    p, t = w["precip_mm"], w["t2m_c"]
    f = pd.DataFrame(index=w.index)
    p30, c30 = p.rolling(30).mean(), prior_years_climatology(p, 30)
    f["precipitation_anomaly_pct"] = 100 * (p30 / c30.clip(lower=0.1) - 1)
    # monsoon (1 Jun - 30 Sep): cumulative rain since 1 June vs the prior-years cumulative to that date
    season = (w.index.month >= 6) & (w.index.month <= 9)
    yr = w.index.year
    cum = p.where(season, 0.0).groupby(yr).cumsum()
    cum_clim = prior_years_climatology(cum.where(season), 1)
    f["monsoon_cumulative_deviation"] = np.where(season, 100 * (cum / cum_clim.clip(lower=1.0) - 1), 0.0)
    f["temperature_anomaly_c"] = t.rolling(7).mean() - prior_years_climatology(t, 7)
    return f


def gpr_features(daily: pd.DataFrame, monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.DataFrame(index=daily.index)
    d["daily_gpr_recent_index"] = np.log(daily["GPRD"].rolling(7).mean())
    d["global_gprt_threats_index"] = np.log(daily["GPRD_THREAT"].rolling(30).mean())
    d["global_gpra_acts_index"] = np.log(daily["GPRD_ACT"].rolling(30).mean())
    ma30 = daily["GPRD"].rolling(30).mean()
    d["global_gpr_change_1m"] = np.log(ma30 / ma30.shift(30))
    m = pd.DataFrame(index=monthly.index)
    m["country_gpr_index"] = np.log(monthly["GPRC_IND"])
    m["country_gpr_change_1m"] = np.log(monthly["GPRC_IND"]).diff()
    return d, m


def _align_with_lag(f: pd.DataFrame, sessions: pd.DatetimeIndex, available_at: pd.DatetimeIndex,
                    max_stale: int) -> pd.DataFrame:
    """Re-date each observation to when it became available, then take the latest one known at each session."""
    f = f.set_axis(available_at)
    return pd.DataFrame({c: align_to_sessions(f[c].rename(c), sessions, "same_session", max_stale)
                         for c in f.columns}, index=sessions)


GEO_GLOBAL = ["daily_gpr_recent_index", "global_gprt_threats_index", "global_gpra_acts_index", "global_gpr_change_1m"]
GEO_INDIA = ["country_gpr_index", "country_gpr_change_1m"]
WEATHER_COLS = ["precipitation_anomaly_pct", "monsoon_cumulative_deviation", "temperature_anomaly_c"]
INDIA_EXCHANGES = {"NSE", "BSE"}


def forecast_groups(sessions: pd.DatetimeIndex, exchange_code: str) -> tuple[dict, list[str]]:
    """Feature groups for the dashboard's forecast engine: {group: frame aligned to sessions} plus notes.
    India listings get global + India GPR and India weather; others get global GPR only (Indian weather
    says nothing about, say, a US stock). Any data failure drops the group with a note instead of failing."""
    notes, groups = [], {}
    india = exchange_code in INDIA_EXCHANGES
    try:
        f, _ = round2_features(sessions, str(sessions[-1].date()), live=True, weather=india)
        groups["geopolitics"] = f[GEO_GLOBAL + (GEO_INDIA if india else [])]
        if india:
            groups["weather"] = f[WEATHER_COLS]
    except Exception as e:     # network / parsing problems must not break the dashboard
        notes.append(f"Geopolitics/weather data unavailable ({type(e).__name__}); those models are skipped.")
    if not india:
        notes.append("Weather models use India-wide weather and are only offered for NSE/BSE listings.")
    return groups, notes


def round2_features(sessions: pd.DatetimeIndex, end: str, synthetic: bool = False, seed: int = 42,
                    live: bool = False, weather: bool = True):
    """Geopolitics + weather features aligned to index sessions. Returns (features, raw frames)."""
    if synthetic:
        rng = np.random.default_rng(seed + 2)
        cols = GEO_GLOBAL + GEO_INDIA + WEATHER_COLS
        f = pd.DataFrame(rng.standard_normal((len(sessions), len(cols))), index=sessions, columns=cols)
        return f.rolling(10, min_periods=1).mean(), {}
    daily, monthly = load_gpr(live=live)
    gd, gm = gpr_features(daily, monthly)
    month_end = gm.index + pd.offsets.MonthEnd(0)
    parts = [
        _align_with_lag(gd, sessions, gd.index + pd.Timedelta(days=GPR_DAILY_LAG_DAYS), max_stale=10),
        _align_with_lag(gm, sessions, month_end + pd.Timedelta(days=GPR_MONTHLY_LAG_DAYS), max_stale=30),
    ]
    raw = {"gpr_daily": daily, "gpr_monthly": monthly}
    if weather:
        w = load_weather(end, live)
        wf = weather_features_daily(w)
        parts.append(_align_with_lag(wf, sessions, wf.index + pd.Timedelta(days=WEATHER_LAG_DAYS), max_stale=10))
        raw["weather_all_india"] = w
    return pd.concat(parts, axis=1), raw
