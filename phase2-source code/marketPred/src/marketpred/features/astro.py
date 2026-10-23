"""Category H - astronomical cycles (EXPLICITLY EXPERIMENTAL HYPOTHESIS).

Predefined variables, fixed before evaluation (no searching over planets/aspects/orbs):
  * lunar phase (illuminated fraction, phase angle sin/cos, +-3 days of new/full moon),
    motivated by reported lunar-cycle effects (Dichev & Janes 2003; Yuan, Zheng & Zhu 2006);
  * Mercury retrograde (popular claim; no peer-reviewed support found);
  * eclipse proximity (+-7 days) and days to the nearest eclipse;
  * Jupiter-Saturn angular separation (sin/cos);
  * count of "hard aspects" (0/90/180 deg, 8 deg orb) among Sun..Saturn.

Conventions: geocentric ecliptic longitudes (J2000 equinox) from PyEphem (VSOP87/ELP), evaluated at
each decision timestamp (UTC). Eclipses are detected as syzygies with the Moon's ecliptic latitude
below 1.5 deg (solar) / 1.0 deg (lunar, umbral) - an approximation verified against published dates
in tests/test_features.py. These quantities are deterministic and known in advance (no leakage).
PyEphem is LGPL-3.0; Swiss Ephemeris (AGPL-3.0 or commercial) is not required.
"""

from __future__ import annotations

import math
from functools import lru_cache

import ephem
import numpy as np
import pandas as pd

from .base import FeatureContext

CAT = "astro"
SYNODIC_DAYS = 29.530588
PLANETS = {"sun": ephem.Sun, "mercury": ephem.Mercury, "venus": ephem.Venus, "mars": ephem.Mars,
           "jupiter": ephem.Jupiter, "saturn": ephem.Saturn}
SOLAR_LAT_LIMIT, LUNAR_LAT_LIMIT = math.radians(1.5), math.radians(1.0)


def ecl_lon(body_cls, when: ephem.Date) -> float:
    return float(ephem.Ecliptic(body_cls(when)).lon)


def angle_diff(a: float, b: float) -> float:
    """Smallest separation between two angles (radians), in [0, pi]."""
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


@lru_cache(maxsize=4)
def eclipse_dates(start_year: int, end_year: int) -> tuple[tuple[pd.Timestamp, str], ...]:
    """Approximate solar and lunar eclipses between the given years."""
    out = []
    d = ephem.Date(f"{start_year}/1/1")
    end = ephem.Date(f"{end_year + 1}/1/1")
    while d < end:
        nm, fm = ephem.next_new_moon(d), ephem.next_full_moon(d)
        nxt = min(nm, fm)
        lat = abs(float(ephem.Ecliptic(ephem.Moon(nxt)).lat))
        if nxt == nm and lat < SOLAR_LAT_LIMIT:
            out.append((pd.Timestamp(nxt.datetime()), "solar"))
        if nxt == fm and lat < LUNAR_LAT_LIMIT:
            out.append((pd.Timestamp(nxt.datetime()), "lunar"))
        d = ephem.Date(nxt + 1)
    return tuple(out)


def compute(times_utc: pd.DatetimeIndex) -> pd.DataFrame:
    """Astronomical features at the given UTC instants."""
    times = pd.DatetimeIndex(times_utc).tz_convert("UTC").tz_localize(None)
    ecl = eclipse_dates(times.min().year - 1, times.max().year + 1)
    ecl_t = np.array([t.value for t, _ in ecl], dtype="int64")
    rows = []
    for t in times:
        when = ephem.Date(t.to_pydatetime())
        moon = ephem.Moon(when)
        prev_new = ephem.previous_new_moon(when)
        age = float(when - prev_new)                      # days since new moon
        phase = 2 * math.pi * age / SYNODIC_DAYS
        to_full = min(abs(float(ephem.next_full_moon(when) - when)),
                      abs(float(when - ephem.previous_full_moon(when))))
        to_new = min(age, abs(float(ephem.next_new_moon(when) - when)))
        lons = {k: ecl_lon(c, when) for k, c in PLANETS.items()}
        merc_prev = ecl_lon(ephem.Mercury, ephem.Date(when - 1))
        dmerc = (lons["mercury"] - merc_prev + math.pi) % (2 * math.pi) - math.pi
        names = list(lons)
        hard = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sep = math.degrees(angle_diff(lons[names[i]], lons[names[j]]))
                hard += any(abs(sep - a) <= 8 for a in (0, 90, 180))
        js = lons["jupiter"] - lons["saturn"]
        days_to_ecl = np.abs(ecl_t - t.value).min() / 86_400e9 if len(ecl_t) else np.nan
        rows.append({
            "moon_illum": float(moon.moon_phase),
            "moon_phase_sin": math.sin(phase), "moon_phase_cos": math.cos(phase),
            "near_new_moon": float(to_new <= 3), "near_full_moon": float(to_full <= 3),
            "mercury_retrograde": float(dmerc < 0),
            "eclipse_window": float(days_to_ecl <= 7), "days_to_eclipse": float(min(days_to_ecl, 60)),
            "jup_sat_sin": math.sin(js), "jup_sat_cos": math.cos(js),
            "hard_aspects": float(hard),
        })
    return pd.DataFrame(rows, index=times_utc)


def build(ctx: FeatureContext) -> pd.DataFrame:
    feats = compute(pd.DatetimeIndex(ctx.decision_ts.to_numpy()))
    feats.index = ctx.dates
    feats.columns = [ctx.register(CAT, c, feats[c], "PyEphem", "deterministic ephemeris") for c in feats.columns]
    return feats
