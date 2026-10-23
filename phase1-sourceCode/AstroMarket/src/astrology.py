"""Astrological interpretation layer, built on the astronomical positions in `ephemeris.py`.

Signs, nakshatras, tithis, aspects and orbs are conventions of astrological traditions, not
physical measurements. Nothing here assigns market meaning (bullish/bearish) to any
configuration; that is left to the statistics in `analysis.py`, which test it against data.
"""
from itertools import combinations

import numpy as np
import pandas as pd

from .astro_features import NAKSHATRAS, NO_RETRO, RASHIS
from .ephemeris import BODIES, AstroSettings, angle_diff, moon_illumination, positions

SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
         "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
ASPECTS = {"conjunction": 0, "sextile": 60, "square": 90, "trine": 120, "opposition": 180}
TITHI_NAMES = ["Pratipada", "Dvitiya", "Tritiya", "Chaturthi", "Panchami", "Shashthi", "Saptami",
               "Ashtami", "Navami", "Dashami", "Ekadashi", "Dvadashi", "Trayodashi", "Chaturdashi"]
PHASES = ["New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous",
          "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"]
SLOW = ["jupiter", "saturn", "uranus", "neptune", "pluto", "rahu", "ketu"]
ASPECT_BODIES = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn",
                 "uranus", "neptune", "pluto", "rahu"]   # ketu is always opposite rahu


def sign_name(lon: float, zodiac: str) -> str:
    i = int(lon // 30) % 12
    return f"{RASHIS[i]} ({SIGNS[i]})" if zodiac == "sidereal" else SIGNS[i]


def nakshatra(lon: float) -> tuple[str, int]:
    span = 360 / 27
    i = int(lon // span) % 27
    return NAKSHATRAS[i], int((lon % span) // (span / 4)) + 1


def tithi(elongation: float) -> dict:
    i = int((elongation % 360) // 12)          # 0..29
    paksha = "Shukla" if i < 15 else "Krishna"
    j = i % 15
    name = "Purnima" if i == 14 else "Amavasya" if i == 29 else TITHI_NAMES[j]
    return {"tithi": i + 1, "tithi_name": name, "paksha": paksha}


def phase_name(elongation: float) -> str:
    return PHASES[int(((elongation % 360) + 22.5) // 45) % 8]


def chart(ts_utc, s: AstroSettings) -> dict:
    """Full chart for one instant: astronomical positions + astrological labels + aspects."""
    pos = positions(ts_utc, s).drop(columns="ts_utc")
    pos["sign"] = [sign_name(l, s.zodiac) for l in pos.longitude]
    pos["degree_in_sign"] = pos.longitude % 30
    nk = [nakshatra(l) for l in pos.longitude]
    pos["nakshatra"] = [n for n, _ in nk]
    pos["pada"] = [p for _, p in nk]
    pos["retrograde"] = [(sp < 0) and b not in NO_RETRO for b, sp in zip(pos.body, pos.speed_deg_day)]
    lon = pos.set_index("body")["longitude"]
    spd = pos.set_index("body")["speed_deg_day"]
    elong = (lon["moon"] - lon["sun"]) % 360
    lunar = {"elongation": elong, "illumination": moon_illumination(ts_utc), "phase": phase_name(elong),
             **tithi(elong), "moon_sign": sign_name(lon["moon"], s.zodiac),
             "moon_nakshatra": nakshatra(lon["moon"])[0]}
    asp = []
    for a, b in combinations(ASPECT_BODIES, 2):
        sep = float(angle_diff(lon[a], lon[b]))
        for name, ang in ASPECTS.items():
            orb = abs(sep - ang)
            if orb <= s.orb:
                # applying if the separation is moving toward the exact angle
                sep_next = float(angle_diff(lon[a] + spd[a] / 24, lon[b] + spd[b] / 24))
                asp.append({"body1": a, "body2": b, "aspect": name, "orb": round(orb, 2),
                            "applying": abs(sep_next - ang) < orb})
    return {"positions": pos, "lunar": lunar,
            "aspects": pd.DataFrame(asp, columns=["body1", "body2", "aspect", "orb", "applying"])}


def sky_table(timestamps, s: AstroSettings, index=None) -> pd.DataFrame:
    """Wide table, one row per timestamp: longitude/speed/sign/retro per body + lunar data.
    `index` (e.g. session dates) replaces the UTC timestamps as the row label."""
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    long = positions(ts, s)
    lon = long.pivot(index="ts_utc", columns="body", values="longitude")[BODIES]
    spd = long.pivot(index="ts_utc", columns="body", values="speed_deg_day")[BODIES]
    c = {"ts_utc": lon.index}
    for b in BODIES:
        c[f"{b}_lon"] = lon[b].to_numpy()
        c[f"{b}_speed"] = spd[b].to_numpy()
        c[f"{b}_sign"] = (lon[b].to_numpy() // 30).astype(int)
        if b not in NO_RETRO:
            c[f"{b}_retro"] = (spd[b].to_numpy() < 0).astype(int)
    elong = ((lon["moon"] - lon["sun"]) % 360).to_numpy()
    c["elongation"] = elong
    c["tithi"] = (elong // 12).astype(int) + 1
    c["paksha"] = np.where(c["tithi"] <= 15, "Shukla", "Krishna")
    c["moon_phase"] = [phase_name(e) for e in elong]
    c["moon_nakshatra"] = (lon["moon"].to_numpy() // (360 / 27)).astype(int)
    for a, b in combinations(ASPECT_BODIES, 2):
        sep = angle_diff(lon[a].to_numpy(), lon[b].to_numpy())
        for name, ang in ASPECTS.items():
            dist = np.abs(sep - ang)
            within = dist <= s.orb
            c[f"asp_{a}_{b}_{name}"] = within.astype(int)
            # Continuous companion (0 outside orb, 1 = exact alignment) - mirrors
            # astro_features.py's orb_... features, kept consistent across both code paths.
            c[f"orb_{a}_{b}_{name}"] = np.where(within, 1 - dist / s.orb, 0.0)
    t = pd.DataFrame(c, index=lon.index)
    if index is not None:
        t.index = pd.DatetimeIndex(index)
    t.index.name = "date"
    return t


def readable(t: pd.DataFrame, zodiac: str) -> pd.DataFrame:
    """Human-readable labels for a sky table (signs, nakshatra names, retro lists, aspects)."""
    names = SIGNS if zodiac == "tropical" else RASHIS
    out = pd.DataFrame(index=t.index)
    for b in ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn"]:
        out[f"{b}_sign"] = [names[i] for i in t[f"{b}_sign"]]
    out["moon_nakshatra"] = [NAKSHATRAS[i] for i in t["moon_nakshatra"]]
    out["tithi"] = t["tithi"]
    out["paksha"] = t["paksha"]
    out["moon_phase"] = t["moon_phase"]
    retro_cols = [c for c in t.columns if c.endswith("_retro")]
    out["retrograde"] = t[retro_cols].apply(
        lambda r: ", ".join(c[:-6].title() for c in retro_cols if r[c]) or "-", axis=1)
    asp_cols = [c for c in t.columns if c.startswith("asp_") and "_moon_" not in c and not c.startswith("asp_moon")]
    out["major_aspects"] = t[asp_cols].apply(
        lambda r: ", ".join(_asp_label(c) for c in asp_cols if r[c]) or "-", axis=1)
    return out


def _asp_label(col: str) -> str:
    _, a, b, kind = col.split("_")
    return f"{a.title()} {kind} {b.title()}"


# ---------------------------------------------------------------- numeric features for models
# Uranus/Neptune/Pluto move so little in any one instrument's history that their sin/cos encoding
# is nearly collinear WITH EACH OTHER (confirmed empirically: 0.94-0.99 correlated across bodies,
# not just within one) - they add no separable signal as model features, only as UI/event display
# (ASPECT_BODIES, detect_events() below are untouched, so "Neptune trine X" still shows in the sky).
REDUNDANT_SLOW_BODIES = {"uranus", "neptune", "pluto"}
# Jupiter/Saturn/the lunar node barely move day-to-day at 1-20 session horizons (<1% of their own
# range) - the position is a near-constant, but a retrograde FLIP is a real, fast event, so only
# the flag is kept for these, not the near-frozen longitude.
NEAR_STATIC_LONGITUDE_BODIES = {"jupiter", "saturn", "rahu"}
FEATURE_PAIR_BODIES = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn"]


def encode_features(t: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Numeric model features from a sky table. Returns (features, slow_feature_names).
    Slow-body features change over years, so they mostly encode 'which era is it' and are
    tracked separately to detect calendar-proxy effects."""
    f = {}
    for b in BODIES:
        if b == "ketu" or b in REDUNDANT_SLOW_BODIES:
            continue                                   # redundant with rahu / with each other
        if b not in NEAR_STATIC_LONGITUDE_BODIES:
            rad = np.deg2rad(t[f"{b}_lon"].to_numpy(float))
            f[f"{b}_sin"], f[f"{b}_cos"] = np.sin(rad), np.cos(rad)
        if f"{b}_retro" in t:
            f[f"{b}_retro"] = t[f"{b}_retro"].to_numpy()
    e = np.deg2rad(t["elongation"].to_numpy(float))
    f["moon_phase_sin"], f["moon_phase_cos"] = np.sin(e), np.cos(e)
    f["waxing"] = (t["paksha"] == "Shukla").astype(int).to_numpy()
    # One continuous separation per pair, not the catalog's 5 near-duplicate named "orb" distances
    # (each is a fixed shift of this same angle, corr ~1.0 with it in practice - see
    # src/catalog_features.py's astro_aspects for the same fix). Sun-moon skipped: it's the same
    # angle moon_phase_sin/cos already encodes.
    for i, a in enumerate(FEATURE_PAIR_BODIES):
        for b in FEATURE_PAIR_BODIES[i + 1:]:
            if {a, b} == {"sun", "moon"}:
                continue
            f[f"{a}_{b}_sep_deg"] = angle_diff(t[f"{a}_lon"].to_numpy(float), t[f"{b}_lon"].to_numpy(float))
    f = pd.DataFrame(f, index=t.index)
    slow = [c for c in f.columns if any(tok in SLOW for tok in c.split("_"))]
    return f, slow


# ---------------------------------------------------------------- events
def detect_events(t: pd.DataFrame, zodiac: str) -> pd.DataFrame:
    """Events between consecutive rows of a sky table (dated at the first row after the event).
    Resolution equals the sampling step, so use daily samples for day-level dates."""
    names = SIGNS if zodiac == "tropical" else RASHIS
    ev = []
    idx = t.index
    for b in BODIES:
        if b == "ketu":
            continue
        s = t[f"{b}_sign"].to_numpy()
        for i in np.nonzero(np.diff(s))[0] + 1:
            if b != "moon":
                ev.append((idx[i], "ingress", b, None, f"{b.title()} enters {names[s[i]]}"))
        if f"{b}_retro" in t:
            r = t[f"{b}_retro"].to_numpy()
            for i in np.nonzero(np.diff(r))[0] + 1:
                kind = "station_retrograde" if r[i] else "station_direct"
                ev.append((idx[i], kind, b, None, f"{b.title()} turns {'retrograde' if r[i] else 'direct'}"))
    e = t["elongation"].to_numpy()
    for i in range(1, len(e)):
        if e[i] < e[i - 1] - 180:
            ev.append((idx[i], "new_moon", "moon", "sun", "New Moon (Amavasya)"))
        if e[i - 1] < 180 <= e[i]:
            ev.append((idx[i], "full_moon", "moon", "sun", "Full Moon (Purnima)"))
    lon = {b: t[f"{b}_lon"].to_numpy() for b in ASPECT_BODIES}
    for a, b in combinations([x for x in ASPECT_BODIES if x != "moon"], 2):
        sep = angle_diff(lon[a], lon[b])
        for name, ang in ASPECTS.items():
            d = sep - ang
            if ang in (0, 180):   # separation cannot pass through 0/180, it touches: use local minimum
                ad = np.abs(d)
                hits = np.nonzero((ad[1:-1] <= ad[:-2]) & (ad[1:-1] < ad[2:]) & (ad[1:-1] < 2))[0] + 1
            else:
                hits = np.nonzero((np.sign(d[1:]) != np.sign(d[:-1])) & (np.abs(d[1:]) < 3))[0] + 1
            for i in hits:
                ev.append((idx[i], f"exact_{name}", a, b, f"{a.title()} {name} {b.title()} (exact)"))
    node = np.minimum(angle_diff(lon["sun"], lon["rahu"]), angle_diff(lon["sun"], (lon["rahu"] + 180) % 360))
    season = node <= 18
    for i in np.nonzero(np.diff(season.astype(int)) == 1)[0] + 1:
        ev.append((idx[i], "eclipse_season", "sun", "rahu", "Eclipse season begins (Sun within 18° of a node)"))
    out = pd.DataFrame(ev, columns=["date", "event_type", "body1", "body2", "detail"])
    return out.sort_values("date").reset_index(drop=True)
