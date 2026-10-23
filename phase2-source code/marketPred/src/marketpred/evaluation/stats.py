"""Statistical comparison of forecasts under serial dependence and multiple testing."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats


def newey_west_var(d: np.ndarray, lag: int) -> float:
    """HAC (Bartlett kernel) long-run variance of a series."""
    d = d - d.mean()
    n = len(d)
    v = d @ d / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        v += 2 * w * (d[k:] @ d[:-k]) / n
    return float(max(v, 1e-300))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 1) -> dict:
    """DM test of equal predictive accuracy (Diebold & Mariano 1995) with Newey-West HAC variance.

    d = loss_a - loss_b; positive mean => model b is more accurate. Lag = max(h-1, 4*(n/100)^(2/9)).
    Harvey-Leybourne-Newbold small-sample correction and Student-t reference distribution.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    lag = max(horizon - 1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    var = newey_west_var(d, lag)
    dm = d.mean() / np.sqrt(var / n)
    hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    t = dm * hln
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return {"mean_diff": float(d.mean()), "dm_stat": float(t), "p_value": float(p), "hac_lag": lag, "n": n}


def circular_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
    return idx[:n]


def block_bootstrap_ci(stat: Callable[[np.ndarray], float], n: int, block: int = 20, reps: int = 500,
                       alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile CI of a statistic computed on circular-block-bootstrap resamples of row indices."""
    rng = np.random.default_rng(seed)
    full = stat(np.arange(n))
    draws = np.array([stat(circular_block_indices(n, block, rng)) for _ in range(reps)])
    draws = draws[np.isfinite(draws)]
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2]) if len(draws) else (np.nan, np.nan)
    return {"estimate": float(full), "ci_low": float(lo), "ci_high": float(hi),
            "p_boot_le0": float((draws <= 0).mean()) if len(draws) else float("nan")}


def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm (1979) step-down adjusted p-values (family-wise error control)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def kupiec_pof(violations: np.ndarray, alpha: float) -> dict:
    """Kupiec (1995) proportion-of-failures test for VaR exceedances."""
    v = np.asarray(violations, bool)
    n, x = len(v), int(v.sum())
    pi = x / n if n else np.nan
    if x in (0, n):
        lr = -2 * (n * np.log(1 - alpha) if x == 0 else n * np.log(alpha))
    else:
        lr = -2 * ((n - x) * np.log(1 - alpha) + x * np.log(alpha) - (n - x) * np.log(1 - pi) - x * np.log(pi))
    return {"n": n, "violations": x, "rate": float(pi), "expected": alpha, "lr": float(lr),
            "p_value": float(stats.chi2.sf(lr, 1))}
