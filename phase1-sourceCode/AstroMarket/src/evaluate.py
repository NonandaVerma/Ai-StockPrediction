"""Shared evaluation helpers used by the dashboard's forecast engine and risk pipeline."""
import numpy as np


def benjamini_hochberg(pvals, alpha):
    p = np.asarray(pvals)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    q_out = np.empty_like(q)
    q_out[order] = np.minimum(q, 1.0)
    return q_out, q_out <= alpha


def newey_west_lags(n: int, horizon: int = 1) -> int:
    """At least the target horizon (overlapping windows), else the usual 4(n/100)^(2/9) rule."""
    return max(int(horizon), int(np.floor(4 * (n / 100) ** (2 / 9))))


def performance(log_rets, per_year: float = 252) -> dict:
    """CAGR, annualised Sharpe, max drawdown and final multiple of a series of period log returns."""
    x = np.asarray(log_rets, float)
    eq = np.exp(np.cumsum(x))
    yrs = len(x) / per_year
    dd = eq / np.maximum.accumulate(eq) - 1
    return {"cagr": float(eq[-1] ** (1 / yrs) - 1) if yrs > 0 else np.nan,
            "sharpe": float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(per_year)),
            "max_drawdown": float(dd.min()), "final_multiple": float(eq[-1])}
