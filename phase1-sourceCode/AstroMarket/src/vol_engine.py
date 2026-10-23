"""Volatility target for Model Comparison: which data categories improve out-of-sample volatility
forecasts beyond the standard HAR benchmark?

Target for row t (decision after close t): annualised realised volatility over the window sessions
t+gap+1 .. t+gap+h, sqrt(252 * mean(r^2)). Features use data up to close t (see Dataset).
Validation: the same expanding walk-forward as the direction models (never shuffled), embargo of
gap+h rows, every model on identical rows. All fitting (redundancy filter, scaler, ridge penalty via
inner purged CV, log-normal bias correction) happens inside each training fold.
"""
import numpy as np
import pandas as pd

from .evaluate import newey_west_lags
from .risk_eval import fit_predict, hac_mean_test, holm, ols_lognormal_var, qlike
from .validation import walk_forward_splits

ANN = 252
VAR_FLOOR = 1e-8


def realized_vol_target(log_ret: pd.Series, gap: int, horizon: int) -> pd.Series:
    """Annualised RMS volatility of the window after row t (NaN where the window is incomplete)."""
    r2 = log_ret.astype(float) ** 2
    win = r2.rolling(horizon, min_periods=horizon).mean().shift(-(gap + horizon))
    return np.sqrt(ANN * win)


def har_features(log_ret: pd.Series) -> pd.DataFrame:
    r = log_ret.astype(float)
    return pd.DataFrame({f"har_log_rv_{k}d": np.log(r.rolling(k).std() * np.sqrt(ANN)) for k in (5, 20, 60)},
                        index=r.index).replace([np.inf, -np.inf], np.nan)


def ewma_vol(log_ret: pd.Series) -> pd.Series:
    return np.sqrt((log_ret.astype(float) ** 2).ewm(alpha=0.06).mean() * ANN)       # RiskMetrics, lambda 0.94


def candidate_groups(ds) -> list[str]:
    """Groups that can be added to HAR: market features and every catalog category that was built."""
    return [g for g in ["market", *ds.category_keys()] if g in ds.X]


MIN_ROWS_FOR_LIVE_FORECAST = 300   # below this, the fit is too thin to trust a live number


def live_forecast(ds, group: str = "c05") -> dict | None:
    """Expected annualised volatility over `ds`'s target window, fit on every available row and
    predicted for the as-of session - the live counterpart to vol_compare()'s backtest. `None`
    when `group` isn't built for this instrument or there's too little data to fit reliably.

    Unlike the backtest (which only ever reports out-of-sample, walk-forward numbers), this fits
    once on the full history: there is no later data left to holdout from for a single live point,
    so the 80% range is approximate (from this fit's own residual spread, not a separate test set).
    Only call this for a (ds, group) combination already shown to validate out-of-sample - see
    vol_compare()/the Model Comparison -> Volatility page - this function does not re-prove that."""
    if group not in ds.X:
        return None
    gap, h = ds.gap, ds.horizon
    lr = ds.px["log_ret"]
    y = realized_vol_target(lr, gap, h)
    har = har_features(lr)
    allX = pd.concat([har, ds.X[group]], axis=1)
    allX = allX.loc[:, ~allX.columns.duplicated()]
    rows = y.notna() & (y > 0) & allX.notna().all(axis=1)
    if int(rows.sum()) < MIN_ROWS_FOR_LIVE_FORECAST:
        return None
    asof = allX.iloc[[-1]]
    if asof.isna().any(axis=1).iloc[0]:
        return None
    idx = y.index[rows]
    z = np.log(y[rows].to_numpy())

    def _point(cols_df):
        """(predicted log-vol, residual variance) from a ridge fit on every row, for the as-of row."""
        X = cols_df.loc[idx].to_numpy(float)
        p, info = fit_predict("regression", X, z, cols_df.iloc[[-1]].to_numpy(float), gap + h)
        return float(p[0]), float(info["resid_var"])

    z_group, var_group = _point(allX)
    z_base, var_base = _point(har)
    q = 1.2816   # 80% two-sided normal quantile, applied in log-vol space
    return {
        "group": group,
        # same lognormal bias correction used throughout this file (exp(2*pred + 2*resid_var) is the
        # corrected VARIANCE; sqrt of that is vol) - exp(pred + resid_var) is the equivalent shortcut.
        "expected_vol": float(np.exp(z_group + var_group)),
        "lo": float(np.exp(z_group - q * np.sqrt(var_group))),
        "hi": float(np.exp(z_group + q * np.sqrt(var_group))),
        "baseline_vol": float(np.exp(z_base + var_base)),
        "n_train": int(rows.sum()),
        "as_of": ds.px.index[-1],
    }


def vol_compare(ds, groups: list[str], folds: int) -> dict:
    gap, h = ds.gap, ds.horizon
    lr = ds.px["log_ret"]
    y = realized_vol_target(lr, gap, h)
    har = har_features(lr)
    groups = [g for g in groups if g in ds.X]
    feats = {g: ds.X[g] for g in groups}
    # identical rows for every model: target known and every feature of every tested group present
    allX = pd.concat([har, *feats.values()], axis=1)
    allX = allX.loc[:, ~allX.columns.duplicated()]
    rows = y.notna() & (y > 0) & allX.notna().all(axis=1)
    idx = y.index[rows]
    if len(idx) < 300:
        return {"status": "insufficient", "message": f"only {len(idx)} usable sessions (need 300)"}
    yv = y[rows].to_numpy()
    z = np.log(yv)
    embargo = gap + h
    splits = [(np.asarray(tr), np.asarray(te)) for tr, te in walk_forward_splits(len(idx), folds, embargo)
              if len(tr) >= 250]
    test_rows = np.concatenate([te for _, te in splits])

    def ridge(cols_df):
        X = cols_df.loc[idx].to_numpy(float)
        pv = np.full(len(idx), np.nan)
        for tr, te in splits:
            p, info = fit_predict("regression", X[tr], z[tr], X[te], embargo)
            pv[te] = np.exp(2 * p + 2 * info["resid_var"])
        return pv

    naive20 = np.exp(har["har_log_rv_20d"].loc[idx].to_numpy())
    preds = {"Naive: trailing 20-day vol": naive20 ** 2,
             "EWMA (RiskMetrics)": ewma_vol(lr).loc[idx].to_numpy() ** 2,
             "HAR (OLS)": ols_lognormal_var(har.loc[idx].to_numpy(float), z, splits),
             "HAR (ridge) - base": ridge(har)}
    labels = {"market": "Market features (dashboard set)"}
    from .catalog_features import CATEGORIES
    for g in groups:
        preds[f"HAR + {labels.get(g) or CATEGORIES.get(g, g)}"] = ridge(pd.concat([har, feats[g]], axis=1))
    if len(groups) > 1:
        preds["HAR + all selected groups"] = ridge(allX)

    true_var = yv ** 2
    base = "HAR (ridge) - base"
    te = test_rows
    lags = newey_west_lags(len(te), embargo)
    q_base = qlike(true_var[te], np.maximum(preds[base][te], VAR_FLOOR))
    out_rows, cand_p = [], []
    actual_up = yv[te] > naive20[te]
    for name, pv in preds.items():
        pvt = np.maximum(pv[te], VAR_FLOOR)
        q = qlike(true_var[te], pvt)
        lv = 0.5 * np.log(pvt)
        err = z[te] - lv
        sst = np.sum((z[te] - z[te].mean()) ** 2)
        row = {"model": name, "qlike": float(q.mean()),
               "qlike_vs_base_pct": float(100 * (q_base.mean() - q.mean()) / q_base.mean()),
               "rmse_log_vol": float(np.sqrt(np.mean(err ** 2))), "r2_log_vol": float(1 - np.sum(err ** 2) / sst),
               "corr": float(np.corrcoef(lv, z[te])[0, 1])}
        if name.startswith("Naive"):
            row["vol_up_down_accuracy"] = np.nan
        else:
            row["vol_up_down_accuracy"] = float(np.mean((np.sqrt(pvt) > naive20[te]) == actual_up))
        if name.startswith("HAR +"):
            dm = hac_mean_test(q_base - q, lags)
            row["dm_p_vs_base"] = dm["p_one_sided"]
            cand_p.append(name)
        else:
            row["dm_p_vs_base"] = np.nan
        out_rows.append(row)
    table = pd.DataFrame(out_rows)
    m = table.model.isin(cand_p)
    table.loc[m, "dm_p_holm"] = holm(table.loc[m, "dm_p_vs_base"].to_numpy())
    chart = pd.DataFrame({"actual": yv[te], "HAR base": np.sqrt(preds[base][te])}, index=idx[te])
    cand = table[m].sort_values("qlike")
    if len(cand):
        best = cand.iloc[0].model
        chart[best] = np.sqrt(preds[best][te])
    up_rate = float(actual_up.mean())
    return {"status": "ok", "table": table, "chart": chart, "n": int(len(te)), "n_independent": int(len(te) // h),
            "eval_start": str(idx[te].min().date()), "eval_end": str(idx[te].max().date()),
            "vol_up_majority_rate": max(up_rate, 1 - up_rate), "hac_lags": lags}
