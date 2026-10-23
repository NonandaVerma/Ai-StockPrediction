"""Astronomical layer: geocentric planetary positions from the Swiss Ephemeris.

Everything here is a physical/astronomical quantity (longitude, latitude, distance, speed,
declination, lunar elongation and illumination). Astrological interpretation (signs,
nakshatras, tithis, aspects) lives in `src/astrology.py`.

The only "astrological" input here is the zodiac frame: tropical longitudes are measured from
the March equinox; sidereal longitudes subtract an ayanamsa (a convention, not a measurement).
"""
import threading
from dataclasses import dataclass

import numpy as np
import pandas as pd
import swisseph as swe

BODY_IDS = {
    "sun": swe.SUN, "moon": swe.MOON, "mercury": swe.MERCURY, "venus": swe.VENUS, "mars": swe.MARS,
    "jupiter": swe.JUPITER, "saturn": swe.SATURN, "uranus": swe.URANUS, "neptune": swe.NEPTUNE,
    "pluto": swe.PLUTO,
}
BODIES = list(BODY_IDS) + ["rahu", "ketu"]   # rahu/ketu = lunar north/south node
AYANAMSAS = {"lahiri": swe.SIDM_LAHIRI, "raman": swe.SIDM_RAMAN,
             "krishnamurti": swe.SIDM_KRISHNAMURTI, "fagan_bradley": swe.SIDM_FAGAN_BRADLEY}

_LOCK = threading.Lock()   # pyswisseph keeps global state (sidereal mode, ephemeris path)


@dataclass(frozen=True)
class AstroSettings:
    zodiac: str = "sidereal"          # sidereal | tropical
    ayanamsa: str = "lahiri"
    node: str = "mean"                # mean | true
    orb: float = 6.0                  # aspect orb in degrees (interpretation layer)
    ephemeris_path: str | None = None

    @classmethod
    def from_cfg(cls, cfg, **override):
        a, f = cfg["astro"], cfg.get("forecast", {}) or {}
        kw = dict(zodiac=a["zodiac"], ayanamsa=a["ayanamsa"], node=f.get("node", "mean"),
                  orb=float(a["aspect_orb_deg"]), ephemeris_path=a.get("ephemeris_path"))
        kw.update({k: v for k, v in override.items() if v is not None})
        return cls(**kw)

    @property
    def ayanamsa_label(self) -> str:
        return self.ayanamsa if self.zodiac == "sidereal" else "none"

    @property
    def ephemeris(self) -> str:
        return "swieph" if self.ephemeris_path else "moshier"


def julian_day(ts) -> float:
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    hours = ts.hour + ts.minute / 60 + ts.second / 3600
    return swe.julday(ts.year, ts.month, ts.day, hours)


def _base_flags(s: AstroSettings) -> int:
    if s.ephemeris_path:
        swe.set_ephe_path(s.ephemeris_path)
        return swe.FLG_SWIEPH | swe.FLG_SPEED
    return swe.FLG_MOSEPH | swe.FLG_SPEED   # built-in Moshier ephemeris, ~1 arcsec, no files needed


def positions(timestamps, s: AstroSettings) -> pd.DataFrame:
    """Long table: one row per (ts_utc, body). Accepts one timestamp or many."""
    ts_index = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(timestamps)))
    ts_index = ts_index.tz_localize("UTC") if ts_index.tz is None else ts_index.tz_convert("UTC")
    node_id = swe.MEAN_NODE if s.node == "mean" else swe.TRUE_NODE
    rows = []
    with _LOCK:
        base = _base_flags(s)
        ecl = base
        if s.zodiac == "sidereal":
            swe.set_sid_mode(AYANAMSAS[s.ayanamsa])
            ecl |= swe.FLG_SIDEREAL
        equ = base | swe.FLG_EQUATORIAL
        for ts in ts_index:
            jd = julian_day(ts)
            for body, pid in list(BODY_IDS.items()) + [("rahu", node_id)]:
                p, _ = swe.calc_ut(jd, pid, ecl)
                q, _ = swe.calc_ut(jd, pid, equ)
                rows.append((ts, body, p[0] % 360.0, p[1], p[2], p[3], q[1]))
            lon, lat, dist, spd, dec = rows[-1][2:]
            rows.append((ts, "ketu", (lon + 180.0) % 360.0, -lat, dist, spd, -dec))
    return pd.DataFrame(rows, columns=["ts_utc", "body", "longitude", "latitude", "distance_au",
                                       "speed_deg_day", "declination"])


def moon_illumination(ts) -> float:
    """Illuminated fraction of the Moon's disc (0 = new, 1 = full)."""
    with _LOCK:
        attr = swe.pheno_ut(julian_day(ts), swe.MOON, swe.FLG_MOSEPH)
    return float(attr[1])


def angle_diff(a, b):
    """Smallest separation between two longitudes, 0..180 degrees."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360.0
    return np.minimum(d, 360.0 - d)
