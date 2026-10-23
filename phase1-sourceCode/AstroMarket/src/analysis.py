"""Historical astrology-market analysis: condition statistics, event studies and analogues.

Everything here describes *observed historical association*. A low p-value after multiple-testing
correction means a pattern is unlikely to be pure chance in this sample; it does not show that
planets cause price moves (seasonality, regime changes and data-mining are alternative explanations).
"""
import numpy as np
import pandas as pd
from scipy import stats

from .astro_features import NAKSHATRAS, RASHIS
from .astrology import SIGNS
from .evaluate import benjamini_hochberg

RETRO_BODIES = ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"]
ANALOGUE_BASES = ["sun", "moon", "mercury", "venus", "mars"]


def forward_returns(log_ret: pd.Series, gap: int, horizon: int) -> pd.Series:
    """Log return over sessions t+gap+1 .. t+gap+horizon, aligned to row t."""
    cum = log_ret.fillna(0).cumsum()
    out = cum.shift(-(gap + horizon)) - cum.shift(-gap)
    out[log_ret.shift(-(gap + horizon)).isna()] = np.nan
    return out


def conditions(sky: pd.DataFrame, zodiac: str) -> pd.DataFrame:
    """Readable categorical conditions per row of a sky table."""
    names = SIGNS if zodiac == "tropical" else RASHIS
    c = {"Moon phase": sky["moon_phase"].to_numpy(),
         "Paksha": sky["paksha"].to_numpy(),
         "Tithi": sky["tithi"].astype(str).to_numpy(),
         "Moon sign": [names[i] for i in sky["moon_sign"]],
         "Moon nakshatra": [NAKSHATRAS[i] for i in sky["moon_nakshatra"]],
         "Sun sign": [names[i] for i in sky["sun_sign"]]}
    for b in RETRO_BODIES:
        c[f"{b.title()} retrograde"] = np.where(sky[f"{b}_retro"] == 1, "yes", "no")
    for col in sky.columns:
        if col.startswith("asp_") and "_moon_" not in col and not col.startswith("asp_moon"):
            _, a, b, kind = col.split("_")
            c[f"{a.title()} {kind} {b.title()}"] = np.where(sky[col] == 1, "yes", "no")
    return pd.DataFrame(c, index=sky.index)


def condition_stats(fwd: pd.Series, cond: pd.DataFrame, alpha=0.05, min_n=30,
                    binary_only_yes=True) -> pd.DataFrame:
    """Mean forward return when a condition holds vs when it does not (Welch t-test),
    with Benjamini-Hochberg FDR correction across every test run."""
    fwd = fwd.dropna()
    cond = cond.loc[fwd.index]
    rows = []
    for col in cond.columns:
        values = cond[col].unique()
        if binary_only_yes and set(values) <= {"yes", "no"}:
            values = ["yes"]
        for v in values:
            on = fwd[cond[col] == v]
            off = fwd[cond[col] != v]
            if len(on) < min_n or len(off) < min_n:
                continue
            t, p = stats.ttest_ind(on, off, equal_var=False)
            rows.append({"condition": col, "value": v, "sessions": len(on),
                         "mean_bps": on.mean() * 1e4, "rest_mean_bps": off.mean() * 1e4,
                         "hit_rate": (on > 0).mean(), "vol_bps": on.std() * 1e4,
                         "t_stat": t, "p_value": p})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value"], out["significant_fdr"] = benjamini_hochberg(out["p_value"].fillna(1), alpha)
    return out.sort_values("p_value").reset_index(drop=True)


def event_study(log_ret: pd.Series, event_dates, pre=5, post=10) -> dict:
    """Average cumulative log return from `pre` sessions before to `post` sessions after each event.
    Day 0 is the first session on/after the event date. Compared with unconditional drift."""
    lr = log_ret.fillna(0)
    idx = lr.index
    pos = np.unique(idx.searchsorted(pd.DatetimeIndex(event_dates)))
    pos = pos[(pos - pre >= 1) & (pos + post < len(idx))]
    if len(pos) == 0:
        return {"n": 0}
    cum = lr.cumsum().to_numpy()
    rel = np.arange(-pre, post + 1)
    paths = np.array([cum[p + rel] - cum[p - pre - 1] for p in pos])
    mean = paths.mean(0)
    se = paths.std(0, ddof=1) / np.sqrt(len(pos)) if len(pos) > 1 else np.full(len(rel), np.nan)
    drift = lr.mean() * (rel + pre + 1)
    end_t, end_p = stats.ttest_1samp(paths[:, -1] - drift[-1], 0) if len(pos) > 2 else (np.nan, np.nan)
    return {"n": int(len(pos)), "path": pd.DataFrame({"session": rel, "mean_cum_ret": mean,
                                                      "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se,
                                                      "unconditional": drift}),
            "end_excess": float(mean[-1] - drift[-1]), "end_t": float(end_t), "end_p": float(end_p),
            "event_sessions": idx[pos]}


def analogue_features(sky: pd.DataFrame) -> pd.DataFrame:
    """Fast-moving configuration used to find similar historical skies (slow bodies excluded
    because matching them mostly matches the calendar era)."""
    f = {}
    for b in ANALOGUE_BASES:
        rad = np.deg2rad(sky[f"{b}_lon"].to_numpy(float))
        f[f"{b}_sin"], f[f"{b}_cos"] = np.sin(rad), np.cos(rad)
    e = np.deg2rad(sky["elongation"].to_numpy(float))
    f["phase_sin"], f["phase_cos"] = 2 * np.sin(e), 2 * np.cos(e)       # weight lunar phase higher
    for b in ["mercury", "venus", "mars"]:
        f[f"{b}_retro"] = sky[f"{b}_retro"].to_numpy(float)
    return pd.DataFrame(f, index=sky.index)


def analogues(hist_feat: pd.DataFrame, fwd: pd.Series, target_feat: pd.Series, k=60,
              min_n=30) -> dict:
    """k nearest historical sessions to the target configuration, with their realised forward
    returns, compared against every eligible session (the unconditional distribution)."""
    ok = fwd.dropna().index.intersection(hist_feat.index)
    if len(ok) < min_n:
        return {"status": "insufficient", "n": int(len(ok)),
                "message": f"Only {len(ok)} historical sessions with known outcomes (need {min_n})."}
    X = hist_feat.loc[ok].to_numpy(float)
    d = np.sqrt(((X - target_feat[hist_feat.columns].to_numpy(float)) ** 2).sum(1))
    order = np.argsort(d)[:min(k, len(ok))]
    sel = ok[order]
    r, base = fwd.loc[sel], fwd.loc[ok]
    t, p = stats.ttest_ind(r, base, equal_var=False)
    hit_p = stats.binomtest(int((r > 0).sum()), len(r), float((base > 0).mean())).pvalue
    table = pd.DataFrame({"date": sel, "distance": d[order], "forward_return": r.to_numpy()})
    strong = len(r) >= min_n and p < 0.05
    return {"status": "ok", "n": int(len(r)), "table": table,
            "mean": float(r.mean()), "median": float(r.median()), "hit_rate": float((r > 0).mean()),
            "std": float(r.std()), "q10": float(r.quantile(.1)), "q90": float(r.quantile(.9)),
            "base_mean": float(base.mean()), "base_hit_rate": float((base > 0).mean()),
            "base_std": float(base.std()), "t_stat": float(t), "p_value": float(p), "hit_p_value": float(hit_p),
            "evidence": "statistically distinguishable (p<0.05, single test)" if strong
            else "not distinguishable from all sessions",
            "median_distance": float(np.median(d[order]))}
