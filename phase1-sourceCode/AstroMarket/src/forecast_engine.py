"""Forecast engine: point-in-time datasets, walk-forward validation, model comparison, forecasts.

Target definition (one definition covers every horizon type):
  For a forecast made after the close of session t ("as-of"), skip `gap` sessions and predict
  the log return over the next `horizon` sessions: close(t+gap) -> close(t+gap+horizon).
    next N sessions      gap = 0,        horizon = N
    a custom date D      gap = k - 1,    horizon = 1   (D is the k-th session after as-of)
    a custom range A..B  gap = sessions before A, horizon = sessions in A..B

Features for row t:
  market  returns, volatility, RSI, MA distance, volume - computed from closes up to t only
  astro   the sky at the first session of the target window (t+gap+1). Planetary positions are
          deterministic, so they are legitimately known in advance.

Leakage controls: walk-forward expanding window (never shuffled), an embargo of gap+horizon rows
between train and test, and forecasts that only ever receive prices up to the as-of session.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .analysis import analogue_features, analogues, forward_returns
from .astrology import encode_features, sky_table
from .ephemeris import AstroSettings
from .evaluate import performance
from .exchanges import TradingCalendar, session_time_utc
from .features import technical_features
from .forecast_models import SPEC_BY_NAME
from .risk_eval import block_permute_columns
from .training import TrainSettings, direction_of, predict, usable_rows, walk_forward

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- data preparation
def market_features(px: pd.DataFrame) -> pd.DataFrame:
    f = technical_features(pd.DataFrame({"close": px["price"]}, index=px.index))
    lr = px["log_ret"]
    f["ret_60"] = lr.rolling(60).sum()
    f["ewma_vol"] = np.sqrt((lr ** 2).ewm(alpha=0.06).mean())
    vol = px["volume"].astype(float)
    if (vol.tail(500) > 0).mean() > 0.9:       # many index feeds report zero volume; skip then
        f["volume_ratio"] = np.log((vol + 1) / (vol.rolling(20).mean() + 1))
    return f


def sky_for_sessions(sessions: pd.DatetimeIndex, cal: TradingCalendar, s: AstroSettings,
                     anchor: str = "open") -> pd.DataFrame:
    ts = session_time_utc(cal.ex, sessions, anchor)
    return sky_table(ts, s, index=sessions)


@dataclass
class Dataset:
    px: pd.DataFrame            # point-in-time prices (ends at as-of)
    X: dict                     # group -> DataFrame aligned to px.index
    fwd: pd.Series              # target log return per row (NaN when unknown at as-of)
    target_sessions: pd.Series  # first session of each row's target window
    slow_astro: list
    gap: int
    horizon: int
    notes: list = field(default_factory=list)   # e.g. why geopolitics / weather groups are missing

    def category_keys(self) -> list:
        return [k for k in self.X if k not in ("market", "astro")]

    def _expand(self, groups) -> list:
        """'all_categories' = every catalog-category group available for this instrument."""
        out = []
        for g in groups:
            out += self.category_keys() if g == "all_categories" else [g]
        return out

    def has_groups(self, groups) -> bool:
        if "all_categories" in groups and not self.category_keys():
            return False
        return all(g in self.X for g in self._expand(groups))

    def matrix(self, groups) -> pd.DataFrame:
        groups = self._expand(groups)
        if not groups:
            return self.X["market"].iloc[:, :0]
        m = pd.concat([self.X[g] for g in groups], axis=1)
        return m.loc[:, ~m.columns.duplicated()]          # categories can share a catalog feature

    @property
    def train_mask(self) -> pd.Series:
        return self.fwd.notna() & self.X["market"].notna().all(axis=1)


def build_dataset(px: pd.DataFrame, sky_all: pd.DataFrame, gap: int, horizon: int,
                  external: dict | None = None, notes: list | None = None,
                  category_ctx: dict | None = None) -> Dataset:
    """`sky_all` must be indexed by px.index followed by at least gap+1 projected sessions.
    `external`: optional extra groups ({"geopolitics": frame, "weather": frame}) already aligned
    point-in-time to sessions (each value re-dated to when it was published).
    `category_ctx` ({"exchange", "symbol", "fetch"}): also build one group per catalog category
    (src/catalog_features.py); astro/calendar categories describe the first target session."""
    n = len(px)
    assert sky_all.index[:n].equals(px.index), "sky table must start with the price sessions"
    if len(sky_all) < n + gap + 1:
        raise ValueError("sky table does not extend far enough past the as-of session")
    astro, slow = encode_features(sky_all)
    pos = np.arange(n) + gap + 1
    a = astro.iloc[pos].set_axis(px.index)
    fwd = forward_returns(px["log_ret"], gap, horizon)
    X = {"market": market_features(px), "astro": a,
         **{g: f.reindex(px.index) for g, f in (external or {}).items()}}
    target_sessions = pd.Series(sky_all.index[pos], index=px.index)
    notes = list(notes or [])
    if category_ctx is not None:
        from .catalog_features import build_category_groups
        cats, cat_notes = build_category_groups(
            px, sky_all.iloc[pos].set_axis(px.index), target_sessions, sky_all.index,
            category_ctx["exchange"], category_ctx["symbol"], category_ctx.get("fetch", True))
        X.update({k: v.reindex(px.index) for k, v in cats.items()})
        notes += cat_notes
    return Dataset(px, X, fwd, target_sessions, slow, gap, horizon, notes)


# ---------------------------------------------------------------- validation
def _wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def strategy(oof: pd.DataFrame, horizon: int, cost_bps: float) -> tuple[pd.DataFrame, dict]:
    """Long-or-cash on non-overlapping windows (one decision every `horizon` sessions),
    costs charged on every position change. Buy-and-hold over the same windows for reference."""
    s = oof.iloc[::max(1, horizon)]
    pos = (s["p_up"] > 0.5).astype(float).to_numpy()
    trades = np.abs(np.diff(np.r_[0.0, pos]))
    strat = pos * s["actual"].to_numpy() - trades * cost_bps / 1e4
    res = pd.DataFrame({"strategy": strat, "buy_hold": s["actual"].to_numpy()}, index=s.index)
    per_year = 252 / max(1, horizon)
    return res, {"strategy": {**performance(strat, per_year), "trades": int(trades.sum()),
                              "time_in_market": float(pos.mean())},
                 "buy_hold": performance(res["buy_hold"].to_numpy(), per_year)}


def metrics(oof: pd.DataFrame, horizon: int, cost_bps: float, directional: bool = True,
            neutral_band: float = 0.03) -> dict:
    y, p, e = oof["actual"].to_numpy(), oof["p_up"].to_numpy(), oof["expected"].to_numpy()
    up = y > 0
    out = {"n": int(len(y)), "eval_start": str(oof.index.min().date()), "eval_end": str(oof.index.max().date()),
           "mae": float(np.mean(np.abs(y - e))), "rmse": float(np.sqrt(np.mean((y - e) ** 2))),
           "r2_vs_zero": float(1 - np.sum((y - e) ** 2) / np.sum(y ** 2)),
           "base_up_rate": float(up.mean())}
    if directional:
        pred = p > 0.5
        out.update({"accuracy": float((pred == up).mean()),
                    "precision": float(precision_score(up, pred, zero_division=0)),
                    "recall": float(recall_score(up, pred, zero_division=0)),
                    "f1": float(f1_score(up, pred, zero_division=0)),
                    "auc": float(roc_auc_score(up, p)) if np.std(p) > 0 and 0 < up.mean() < 1 else 0.5,
                    "brier": float(np.mean((p - up) ** 2))})
        # Accuracy over always guessing the more common direction, and on confident calls only
        # (|P(up) - 0.5| > neutral_band): fewer calls, but the ones a user would act on.
        out["accuracy_edge"] = out["accuracy"] - max(out["base_up_rate"], 1 - out["base_up_rate"])
        sure = np.abs(p - 0.5) > neutral_band
        out["confident_share"] = float(sure.mean())
        out["confident_accuracy"] = float((pred[sure] == up[sure]).mean()) if sure.any() else np.nan
        # Significance on NON-overlapping rows (overlapping windows would overstate n).
        s = oof.iloc[::max(1, horizon)]
        hits = int(((s.p_up > 0.5) == (s.actual > 0)).sum())
        base = max(float((s.actual > 0).mean()), 1 - float((s.actual > 0).mean()))
        out["n_independent"] = int(len(s))
        out["edge_p_value"] = float(stats.binomtest(hits, len(s), base, alternative="greater").pvalue) \
            if len(s) else np.nan
        out["accuracy_independent"] = hits / len(s) if len(s) else np.nan
        out["majority_rate"] = base
        _, bt = strategy(oof, horizon, cost_bps)
        out["backtest"] = bt
    else:
        for k in ("accuracy", "precision", "recall", "f1", "auc", "brier", "edge_p_value", "accuracy_edge",
                  "confident_share", "confident_accuracy"):
            out[k] = np.nan
    return out


def evidence_label(m: dict, family: str) -> str:
    if family == "baseline":
        return "baseline"
    if m.get("n_independent", 0) < 30:
        return "insufficient"
    ok = (m.get("edge_p_value", 1) < 0.05) and (m.get("auc", 0.5) > 0.5)
    return "validated" if ok else "experimental"


def compare(ds: Dataset, models: list[str], train: TrainSettings, cost_bps: float) -> dict:
    """Walk-forward every model with the shared training recipe. Returns {model: {"oof", "spec", "metrics"}}.
    Each model is scored on all of its own usable dates, so its scorecard is the same whichever other
    models are selected; head-to-head tests (astro_adds_value) and common_dates() use shared dates."""
    res = {}
    for name in models:
        spec = SPEC_BY_NAME[name]
        if not ds.has_groups(spec.groups):     # e.g. weather for a non-Indian listing
            continue
        oof = walk_forward(ds, name, train)
        if oof.empty:
            continue
        m = metrics(oof, ds.horizon, cost_bps, directional=name != "naive_zero", neutral_band=train.neutral_band)
        m["evidence"] = evidence_label(m, spec.family)
        res[name] = {"oof": oof, "spec": spec, "metrics": m}
    return res


def common_dates(res: dict) -> pd.Index:
    """Dates every model in `res` has an out-of-sample prediction for (fair side-by-side charts)."""
    common = None
    for r in res.values():
        common = r["oof"].index if common is None else common.intersection(r["oof"].index)
    return common if common is not None else pd.Index([])


def astro_adds_value(res: dict, horizon: int) -> list[dict]:
    """Paired tests on identical non-overlapping dates: does adding astro features help?"""
    rows = []
    pairs = [("hybrid_linear", "market_linear"), ("hybrid_lgbm", "market_lgbm"),
             ("astro_stat", "naive_mean"), ("astro_lgbm", "naive_mean"), ("market_linear", "naive_mean")]
    # every "market + category" model against market-only of the same algorithm
    pairs += [(n, f"market_{SPEC_BY_NAME[n].algo}") for n in res
              if SPEC_BY_NAME[n].family in ("category", "geopolitics", "weather")]
    for a, b in pairs:
        if a not in res or b not in res:
            continue
        both = res[a]["oof"].index.intersection(res[b]["oof"].index)          # identical dates for the pair
        oa = res[a]["oof"].loc[both].iloc[::max(1, horizon)]
        ob = res[b]["oof"].loc[both].iloc[::max(1, horizon)]
        ha = (oa.p_up > 0.5) == (oa.actual > 0)
        hb = (ob.p_up > 0.5) == (ob.actual > 0)
        n01, n10 = int((~ha & hb).sum()), int((ha & ~hb).sum())
        p = stats.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0      # exact McNemar
        se_a, se_b = (oa.actual - oa.expected) ** 2, (ob.actual - ob.expected) ** 2
        dm_p = stats.ttest_rel(se_a, se_b).pvalue if len(se_a) > 2 else np.nan       # squared-error comparison
        auc = [roc_auc_score(o.actual > 0, o.p_up) if o.p_up.std() > 0 and 0 < (o.actual > 0).mean() < 1 else 0.5
               for o in (res[a]["oof"].loc[both], res[b]["oof"].loc[both])]
        rows.append({"model": a, "vs": b, "accuracy_diff": float(ha.mean() - hb.mean()),
                     "auc_diff": float(auc[0] - auc[1]),
                     "mcnemar_p": float(p), "rmse_diff": float(np.sqrt(se_a.mean()) - np.sqrt(se_b.mean())),
                     "sq_error_p": float(dm_p), "n_independent": int(len(oa))})
    return rows


def permutation_test(ds: Dataset, model: str, train: TrainSettings, n_perm: int, block=63) -> dict:
    """Shuffle quarter-long blocks of the astro features in time (breaking any sky-return link
    while keeping short-run structure) and re-run validation. p = share of shuffles >= real AUC."""
    rng = np.random.default_rng(train.seed)
    spec = SPEC_BY_NAME[model]
    real = metrics(walk_forward(ds, model, train), ds.horizon, 0)["auc"]
    X = ds.matrix(spec.groups)
    astro = [X.columns.get_loc(c) for c in ds.X["astro"].columns]
    null = []
    for _ in range(n_perm):
        Xs = pd.DataFrame(block_permute_columns(X.to_numpy(float), astro, block, rng), index=X.index, columns=X.columns)
        null.append(metrics(walk_forward(ds, model, train, Xs), ds.horizon, 0)["auc"])
    null = np.array(null)
    return {"real_auc": real, "null": null.tolist(), "null_mean": float(null.mean()),
            "p_value": float((1 + (null >= real).sum()) / (1 + n_perm))}


# ---------------------------------------------------------------- forecasting
def risk_category(px: pd.DataFrame) -> tuple[float, str, float]:
    """EWMA (RiskMetrics, lambda=0.94) daily volatility at as-of and its historical percentile."""
    v = np.sqrt((px["log_ret"] ** 2).ewm(alpha=0.06).mean()).dropna()
    cur = float(v.iloc[-1])
    pct = float((v < cur).mean())
    cat = "low" if pct < 0.33 else "normal" if pct < 0.67 else "elevated" if pct < 0.9 else "high"
    return cur, cat, pct


def calibrated_confidence(oof: pd.DataFrame, p: float, horizon: int, min_n: int) -> dict:
    """Validated hit rate of past out-of-sample predictions made in the same direction with a
    similar probability. Reported only when enough independent cases exist."""
    up = p > 0.5
    band = oof[((oof.p_up > 0.5) == up) & ((oof.p_up - p).abs() <= 0.05)]
    n_eff = len(band) // max(1, horizon)
    if n_eff < min_n:
        return {"calibrated": False, "n": n_eff,
                "note": f"only {n_eff} independent validated cases near this probability (need {min_n})"}
    hits = (band.actual > 0) == up
    k = int(round(hits.mean() * n_eff))
    lo, hi = _wilson(k, n_eff)
    return {"calibrated": True, "hit_rate": float(hits.mean()), "ci_low": lo, "ci_high": hi, "n": n_eff}


def resolve_window(cal: TradingCalendar, as_of, mode: str, horizon: int | None = None,
                   date=None, start=None, end=None) -> dict:
    """Translate a user request into (gap, horizon, window sessions) with explanatory notes."""
    as_of = pd.Timestamp(as_of)
    notes = []
    if mode == "next":
        window = cal.next_sessions(as_of, horizon)
    elif mode == "date":
        m = cal.map_date(date)
        if m["session"] <= as_of:
            raise ValueError("Target date must be after the as-of session.")
        if m["note"] != "trading session":
            notes.append(m["note"])
        window = pd.DatetimeIndex([m["session"]])
    elif mode == "range":
        if pd.Timestamp(end) < pd.Timestamp(start):
            raise ValueError("Range end is before range start.")
        window = cal.sessions_between(max(pd.Timestamp(start), as_of + pd.Timedelta(days=1)), end)
        if len(window) == 0:
            raise ValueError("No trading sessions fall inside the selected range.")
        skipped = len(pd.date_range(start, end)) - len(window)
        if skipped:
            notes.append(f"{skipped} calendar day(s) in the range are weekends/holidays and carry no return.")
    else:
        raise ValueError(mode)
    before = cal.sessions_between(as_of + pd.Timedelta(days=1), window[0] - pd.Timedelta(days=1))
    gap = len(before)
    if gap:
        notes.append(f"{gap} session(s) between the as-of close and the window start are skipped (gap).")
    notes += cal.coverage_notes(window)
    return {"gap": gap, "horizon": len(window), "window": window, "notes": notes}


def forecast(px_pit: pd.DataFrame, cal: TradingCalendar, s: AstroSettings, req: dict, fc_cfg: dict,
             cost_bps: float, models: list[str], primary: str = "astro_stat", train: TrainSettings | None = None,
             external: dict | None = None, external_notes: list | None = None,
             category_ctx: dict | None = None) -> dict:
    """Forecast for one request using ONLY px_pit (prices up to and including the as-of session).
    `external` groups must be aligned to px_pit.index using publication dates (no later information).
    Training, scoring and the 80% range come from src/training.py (same recipe as every other page)."""
    train = train or TrainSettings.from_cfg({"forecast": fc_cfg})
    as_of = px_pit.index[-1]
    gap, h, window = req["gap"], req["horizon"], req["window"]
    future = cal.next_sessions(as_of, gap + h)
    sky_all = sky_for_sessions(px_pit.index.append(future), cal, s, fc_cfg.get("session_anchor", "open"))
    ds = build_dataset(px_pit, sky_all, gap, h, external, external_notes, category_ctx)
    n_train = usable_rows(ds, train)
    out = {"as_of": as_of, "gap": gap, "horizon": h, "window": window, "n_train": n_train,
           "generated_at": pd.Timestamp.now(tz="UTC"), "notes": list(req.get("notes", [])) + ds.notes,
           "last_close": float(px_pit["close"].iloc[-1]), "training": train.describe()}
    if n_train < train.min_train_sessions:
        out["status"] = "unavailable"
        out["message"] = (f"Only {n_train} sessions with known outcomes before {as_of.date()} "
                          f"(minimum {train.min_train_sessions}). No forecast is produced.")
        return out

    vol_d, risk, vol_pct = risk_category(px_pit)
    out.update({"daily_vol": vol_d, "expected_vol": vol_d * np.sqrt(h), "risk": risk, "vol_percentile": vol_pct})

    res = compare(ds, models, train, cost_bps)
    band = train.neutral_band
    rows = []
    for name, r in res.items():
        spec = r["spec"]
        try:
            f = predict(ds, name, train, r["oof"])
        except ValueError as e:
            rows.append({"model": name, "label": spec.label, "family": spec.family, "status": f"unavailable ({e})"})
            continue
        p, e = f["prob_up"], f["expected"]
        met = r["metrics"]
        conf = calibrated_confidence(r["oof"], p, h, int(fc_cfg.get("calibration_min_n", 40))) \
            if name != "naive_zero" else {"calibrated": False, "n": 0, "note": "no directional call"}
        rows.append({"model": name, "label": spec.label, "family": spec.family, "version": spec.version,
                     "prob_up": p, "direction": "neutral" if name == "naive_zero" else direction_of(p, band),
                     "expected_return": e, "interval_low": f["lo"], "interval_high": f["hi"],
                     "evidence": met["evidence"], "confidence": conf, "metrics": met, "n_train": f["n_train"],
                     "selected_features": f["fitted"].selected_features(f["columns"])[:15]})
    for name in models:
        if name not in res:
            rows.append({"model": name, "label": SPEC_BY_NAME[name].label, "family": SPEC_BY_NAME[name].family,
                         "status": "unavailable (data not available for this instrument)"})
    out["models"] = rows
    out["comparison"] = res
    out["astro_vs"] = astro_adds_value(res, h)
    ok = [r for r in rows if "prob_up" in r]
    out["primary"] = next((r for r in ok if r["model"] == primary), ok[0] if ok else rows[0])
    out["status"] = "ok"

    # Historical analogues: past sessions whose sky resembles the first target session, with the
    # return over the following h sessions (only outcomes completed by the as-of close).
    feat = analogue_features(sky_all)
    fwd_from = forward_returns(px_pit["log_ret"], 0, h).shift(1)    # return over sessions j..j+h-1
    out["analogues"] = analogues(feat.loc[px_pit.index], fwd_from, feat.loc[window[0]],
                                 k=int(fc_cfg.get("analogue_k", 60)), min_n=int(fc_cfg.get("min_analogues", 30)))
    out["window_sky"] = sky_all.loc[window]
    return out
