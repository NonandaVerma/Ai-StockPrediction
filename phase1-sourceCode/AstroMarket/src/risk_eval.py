"""Out-of-sample evaluation for the risk pipeline.

Rules (all enforced in code, none tuned on test data):
  * Purged expanding walk-forward: a training row t is used for a test block starting at t0
    only if its label window t+1..t+H ends before t0 (train = rows < t0 - H).
  * Every fitted step - redundancy filter, scaler, ridge penalty (inner purged CV), large-loss
    threshold, VaR quantile - is fitted on the training rows of that fold only.
  * All models are scored on identical rows.
  * Comparisons use Diebold-Mariano tests with Newey-West standard errors (lags >= H, because
    neighbouring 20-day labels overlap), plus a block-permutation test of the added group.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .risk_targets import from_model_scale, to_model_scale

RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
LOSS_QUANTILE = 0.10      # large loss = 20-day return below the training-fold 10th percentile
VAR_LEVEL = 0.05
REDUNDANCY_CORR = 0.95


# ---------------------------------------------------------------- splits
def purged_walk_forward(n: int, horizon: int, min_train: int, test_size: int):
    t0 = min_train + horizon
    while t0 < n:
        t1 = min(n, t0 + test_size)
        yield np.arange(0, t0 - horizon), np.arange(t0, t1)
        t0 = t1


# ---------------------------------------------------------------- fold-level fitting
def redundancy_filter(X: np.ndarray, threshold: float = REDUNDANCY_CORR) -> np.ndarray:
    """Keep a column unless it is |corr| > threshold with an earlier kept column (training rows only)."""
    keep = []
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(X, rowvar=False) if X.shape[1] > 1 else np.ones((1, 1))
    for j in range(X.shape[1]):
        if np.nanstd(X[:, j]) == 0:
            continue
        if all(abs(corr[j, k]) <= threshold for k in keep):
            keep.append(j)
    return np.array(keep, dtype=int)


def _ridge(horizon):
    return make_pipeline(StandardScaler(),
                         RidgeCV(alphas=RIDGE_ALPHAS, cv=TimeSeriesSplit(n_splits=4, gap=horizon)))


def fit_predict(kind, Xtr, ytr, Xte, horizon, fwd_tr=None, fwd_te=None):
    """kind: 'regression' (y on model scale) or 'large_loss'. Returns (pred, info)."""
    keep = redundancy_filter(Xtr)
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    if kind == "regression":
        m = _ridge(horizon).fit(Xtr, ytr)
        fit_tr = m.predict(Xtr)
        resid_var = float(np.var(ytr - fit_tr))
        return m.predict(Xte), {"keep": keep, "resid_var": resid_var, "alpha": float(m[-1].alpha_),
                                "pred_train": fit_tr}
    q = float(np.quantile(fwd_tr, LOSS_QUANTILE))
    ytr_b, yte_b = (fwd_tr < q).astype(int), (fwd_te < q).astype(int)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(Xtr, ytr_b)
    return m.predict_proba(Xte)[:, 1], {"keep": keep, "threshold": q, "y_test": yte_b,
                                        "base_rate": float(ytr_b.mean())}


def oof_predictions(X: np.ndarray, data: dict, kind: str, target: str, horizon: int,
                    min_train: int = 0, test_size: int = 0, splits=None) -> dict:
    """Out-of-fold predictions on the purged walk-forward (or on explicit `splits`, e.g. the
    single dev -> holdout split). `data` holds aligned numpy arrays: y_raw (target on natural
    scale, regression only) and fwd (20d forward return)."""
    n = len(X)
    pred = np.full(n, np.nan)
    extra = np.full(n, np.nan)          # variance correction (regression) or binary label (loss)
    base_rate = np.full(n, np.nan)
    var_q = np.full(n, np.nan)          # training-fold quantile of standardised 20d returns (VaR)
    fold_id = np.full(n, -1)
    keeps = []
    splits = splits if splits is not None else purged_walk_forward(n, horizon, min_train, test_size)
    for k, (tr, te) in enumerate(splits):
        if kind == "regression":
            z = to_model_scale(target, data["y_raw"])
            p, info = fit_predict(kind, X[tr], z[tr], X[te], horizon)
            extra[te] = info["resid_var"]
            if target == "realized_volatility_next_20d":
                vol_tr = np.sqrt(np.exp(2 * info["pred_train"] + 2 * info["resid_var"]))
                var_q[te] = np.quantile(data["fwd"][tr] / (vol_tr * np.sqrt(horizon / 252)), VAR_LEVEL)
        else:
            p, info = fit_predict(kind, X[tr], None, X[te], horizon, data["fwd"][tr], data["fwd"][te])
            extra[te], base_rate[te] = info["y_test"], info["base_rate"]
        pred[te], fold_id[te] = p, k
        keeps.append(info["keep"])
    return {"pred": pred, "extra": extra, "base_rate": base_rate, "var_q": var_q, "fold": fold_id,
            "keeps": keeps}


# ---------------------------------------------------------------- losses
def qlike(true_var, pred_var):
    ratio = true_var / pred_var
    return ratio - np.log(ratio) - 1


def regression_losses(target, oof, y_raw):
    """Per-row losses. Volatility: QLIKE on variance (robust to noisy proxies; Patton 2011) with a
    lognormal mean correction fitted in training; drawdown: squared error on the model scale."""
    z_hat, s2 = oof["pred"], oof["extra"]
    z = to_model_scale(target, y_raw)
    out = {"sq_err_model_scale": (z - z_hat) ** 2}
    if target == "realized_volatility_next_20d":
        pred_var = np.exp(2 * z_hat + 2 * s2)
        out["qlike"] = qlike(y_raw ** 2, pred_var)
        out["primary"] = out["qlike"]
    else:
        out["abs_err"] = np.abs(y_raw - from_model_scale(target, z_hat))
        out["primary"] = out["sq_err_model_scale"]
    return out


def classification_losses(oof):
    p, y = np.clip(oof["pred"], 1e-6, 1 - 1e-6), oof["extra"]
    brier = (p - y) ** 2
    return {"brier": brier, "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)), "primary": brier}


def benchmark_vol_losses(pred_vol, y_raw):
    return {"qlike": qlike(y_raw ** 2, pred_vol ** 2),
            "sq_err_model_scale": (np.log(y_raw) - np.log(pred_vol)) ** 2}


# ---------------------------------------------------------------- tests
def hac_mean_test(d: np.ndarray, lags: int) -> dict:
    """Diebold-Mariano: H0 mean loss differential <= 0. d = loss(base) - loss(candidate), so
    positive mean = candidate better. One-sided p with Newey-West variance."""
    d = np.asarray(d, float)
    n = len(d)
    u = d - d.mean()
    s = u @ u / n
    for lag in range(1, lags + 1):
        s += 2 * (1 - lag / (lags + 1)) * (u[lag:] @ u[:-lag]) / n
    se = np.sqrt(max(s, 1e-300) / n)
    t = d.mean() / se
    return {"mean_diff": float(d.mean()), "t_stat": float(t), "p_one_sided": float(stats.norm.sf(t))}


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def yearly_win_share(d: pd.Series) -> tuple[float, int]:
    by_year = d.groupby(d.index.year).mean()
    return float((by_year > 0).mean()), int(len(by_year))


def block_permute_columns(X: np.ndarray, cols: list[int], block: int, rng) -> np.ndarray:
    """Shuffle blocks of rows of the given columns only: breaks their link to the target and to the
    other features, while keeping their own persistence (a plain row shuffle would not)."""
    n = len(X)
    blocks = [np.arange(i, min(i + block, n)) for i in range(0, n, block)]
    order = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
    Xp = X.copy()
    Xp[:, cols] = X[order][:, cols]
    return Xp


def kupiec_pof(exceed: np.ndarray, level: float) -> dict:
    n, x = len(exceed), int(np.sum(exceed))
    if n == 0:
        return {"n": 0, "exceedances": 0, "rate": np.nan, "p_value": np.nan}
    pi = x / n
    ll0 = (n - x) * np.log(1 - level) + x * np.log(level)
    ll1 = ((n - x) * np.log(1 - pi) if x < n else 0) + (x * np.log(pi) if x > 0 else 0)
    lr = -2 * (ll0 - ll1)
    return {"n": n, "exceedances": x, "rate": pi, "p_value": float(stats.chi2.sf(lr, 1))}


def var_backtest(fwd: np.ndarray, vol_pred: np.ndarray, std_resid_q: np.ndarray, horizon: int) -> dict:
    """20-day VaR = training quantile of standardised 20d returns x predicted 20d vol.
    Tested on NON-overlapping windows (every H-th row) so exceedances are independent."""
    scale = vol_pred * np.sqrt(horizon / 252)
    var = std_resid_q * scale
    ok = ~np.isnan(var) & ~np.isnan(fwd)
    idx = np.flatnonzero(ok)[::horizon]
    return kupiec_pof(fwd[idx] < var[idx], VAR_LEVEL)


def auc_safe(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else np.nan


def ols_lognormal_var(X: np.ndarray, z: np.ndarray, splits) -> np.ndarray:
    """Plain OLS of log volatility `z` on X per split (no penalty, no filter: the textbook HAR
    benchmark), returned as variance with the lognormal correction exp(2 pred + 2 resid_var)."""
    pv = np.full(len(z), np.nan)
    for tr, te in splits:
        m = LinearRegression().fit(X[tr], z[tr])
        pv[te] = np.exp(2 * m.predict(X[te]) + 2 * np.var(z[tr] - m.predict(X[tr])))
    return pv
