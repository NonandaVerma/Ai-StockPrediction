"""Step 2 - Astrological features computed with the Swiss Ephemeris (pyswisseph).

For every trading day we compute the sky at a fixed time (default 09:15 IST, NSE open)
and turn it into numeric features a model can use.

Feature groups produced (column prefixes):
  <planet>_sin / _cos   longitude encoded on a circle (359 deg and 1 deg are neighbours)
  <planet>_retro        1 if the planet is retrograde
  <planet>_sign         rashi / zodiac sign index 0-11
  <planet>_ingress      1 on the first trading day after a sign change
  moon_nakshatra        0-26 lunar mansion
  tithi, paksha         Vedic lunar day 0-29 and fortnight (0 = Shukla/waxing, 1 = Krishna/waning)
  moon_phase_sin/cos    Sun-Moon elongation on a circle
  asp_<p1>_<p2>_<type>  1 if the two bodies form that aspect within the orb
  eclipse_season        1 if the Sun is within 18 deg of Rahu or Ketu
  eclipse_window        eclipse season AND near Amavasya/Purnima
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import swisseph as swe

from .ephemeris import AYANAMSAS, angle_diff

PLANETS = {
    "sun": swe.SUN, "moon": swe.MOON, "mercury": swe.MERCURY, "venus": swe.VENUS,
    "mars": swe.MARS, "jupiter": swe.JUPITER, "saturn": swe.SATURN, "rahu": swe.MEAN_NODE,
}
# Planets that move so slowly their position mostly encodes "which year is it". Neptune/Uranus/Pluto
# aren't in this module's PLANETS set, but are flagged here too for consistency with
# src/catalog_features.py's REDUNDANT_SLOW_BODIES/NEAR_STATIC_LONGITUDE_BODIES, which is where
# they're actually excluded from the model-facing feature catalog.
SLOW_BODIES = {"jupiter", "saturn", "rahu", "ketu", "uranus", "neptune", "pluto"}
NO_RETRO = {"sun", "moon", "rahu", "ketu"}  # Sun/Moon never retrograde; nodes always are

ASPECTS = {"conj": 0, "sext": 60, "square": 90, "trine": 120, "opp": 180}

RASHIS = ["Mesha", "Vrishabha", "Mithuna", "Karka", "Simha", "Kanya",
          "Tula", "Vrishchika", "Dhanu", "Makara", "Kumbha", "Meena"]
NAKSHATRAS = ["Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu",
              "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni", "Hasta",
              "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha",
              "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha", "Purva Bhadrapada",
              "Uttara Bhadrapada", "Revati"]


def _flags(astro_cfg):
    if astro_cfg.ephemeris_path:
        swe.set_ephe_path(astro_cfg.ephemeris_path)
        flags = swe.FLG_SWIEPH | swe.FLG_SPEED
    else:
        flags = swe.FLG_MOSEPH | swe.FLG_SPEED  # built-in, accurate to ~1 arcsec, no files needed
    if astro_cfg.zodiac == "sidereal":
        swe.set_sid_mode(AYANAMSAS[astro_cfg.ayanamsa])
        flags |= swe.FLG_SIDEREAL
    return flags


def positions_on(date: pd.Timestamp, astro_cfg) -> dict:
    """Longitude (deg) and speed (deg/day) of each body for one date."""
    flags = _flags(astro_cfg)
    jd = swe.julday(date.year, date.month, date.day, float(astro_cfg.observation_time_utc))
    out = {}
    for name, pid in PLANETS.items():
        pos, _ = swe.calc_ut(jd, pid, flags)
        out[name] = (pos[0] % 360.0, pos[3])
    lon_r, spd_r = out["rahu"]
    out["ketu"] = ((lon_r + 180.0) % 360.0, spd_r)
    return out


def compute_astro_features(dates: pd.DatetimeIndex, astro_cfg, cache_dir="data/processed") -> pd.DataFrame:
    cache = Path(cache_dir) / (f"astro_{astro_cfg.zodiac}_{astro_cfg.ayanamsa}_"
                               f"{dates.min():%Y%m%d}_{dates.max():%Y%m%d}_{len(dates)}.csv")
    if cache.exists():
        return pd.read_csv(cache, index_col="date", parse_dates=True)

    rows = []
    for d in dates:
        pos = positions_on(d, astro_cfg)
        rows.append({**{f"{k}_lon": v[0] for k, v in pos.items()},
                     **{f"{k}_speed": v[1] for k, v in pos.items()}})
    raw = pd.DataFrame(rows, index=dates)
    raw.index.name = "date"

    bodies = list(PLANETS) + ["ketu"]
    f = {}
    for b in bodies:
        lon = raw[f"{b}_lon"].values
        rad = np.deg2rad(lon)
        f[f"{b}_sin"] = np.sin(rad)
        f[f"{b}_cos"] = np.cos(rad)
        f[f"{b}_sign"] = (lon // 30).astype(int)
        f[f"{b}_ingress"] = (np.diff(f[f"{b}_sign"], prepend=f[f"{b}_sign"][0]) != 0).astype(int)
        if b not in NO_RETRO:
            f[f"{b}_retro"] = (raw[f"{b}_speed"].values < 0).astype(int)

    sun, moon = raw["sun_lon"].values, raw["moon_lon"].values
    elong = (moon - sun) % 360.0
    f["moon_nakshatra"] = (moon // (360.0 / 27)).astype(int)
    f["tithi"] = (elong // 12).astype(int)
    f["paksha"] = (f["tithi"] >= 15).astype(int)
    f["moon_phase_sin"] = np.sin(np.deg2rad(elong))
    f["moon_phase_cos"] = np.cos(np.deg2rad(elong))

    orb = float(astro_cfg.aspect_orb_deg)
    for p1, p2 in combinations(list(PLANETS), 2):  # Ketu skipped: always opposite Rahu
        sep = angle_diff(raw[f"{p1}_lon"].values, raw[f"{p2}_lon"].values)
        for asp, ang in ASPECTS.items():
            dist = np.abs(sep - ang)
            col = (dist <= orb).astype(int)
            # Keep only aspects with enough days both on and off to be statistically
            # testable (matches the n>=30 bar evaluate.single_rule_tests already uses) -
            # rare aspects are mostly noise/dimensionality, not signal.
            if 30 <= col.sum() <= len(col) - 30:
                f[f"asp_{p1}_{p2}_{asp}"] = col
                # Continuous companion: how close to exact (0 outside orb, 1 = exact
                # alignment), instead of only a hard yes/no cutoff.
                f[f"orb_{p1}_{p2}_{asp}"] = np.where(col, 1 - dist / orb, 0.0)

    node_dist = np.minimum(angle_diff(sun, raw["rahu_lon"].values),
                           angle_diff(sun, raw["ketu_lon"].values))
    f["eclipse_season"] = (node_dist <= 18).astype(int)
    near_syzygy = (angle_diff(elong, 0) <= 12) | (angle_diff(elong, 180) <= 12)
    f["eclipse_window"] = (f["eclipse_season"].astype(bool) & near_syzygy).astype(int)

    f = pd.DataFrame(f, index=dates)
    f.index.name = "date"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    f.to_csv(cache)
    return f


def is_slow_feature(col: str) -> bool:
    """True if a feature involves a slow body (a hidden calendar proxy)."""
    return any(tok in SLOW_BODIES for tok in col.split("_"))
