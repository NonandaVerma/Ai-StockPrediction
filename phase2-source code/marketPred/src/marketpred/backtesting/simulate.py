"""Long/cash strategy simulation from 1-day direction probabilities, net of costs.

Pre-specified rule (not optimised): hold the index on day t+1 if P(up) > threshold (default 0.5),
otherwise hold cash earning the 3-month T-bill yield known at t. Costs are charged on every change of
position: (cost_bps + slippage_bps) per unit of turnover. Index returns exclude dividends, which
understates both buy-and-hold and the strategy when invested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def performance(r: pd.Series, rf: pd.Series | None = None) -> dict:
    r = r.dropna()
    ex = r - (rf.reindex(r.index).fillna(0) if rf is not None else 0)
    ann = 252
    eq = (1 + r).cumprod()
    down = ex[ex < 0]
    return {
        "ann_return": float(eq.iloc[-1] ** (ann / len(r)) - 1) if len(r) else float("nan"),
        "ann_vol": float(r.std() * np.sqrt(ann)),
        "sharpe": float(ex.mean() / ex.std() * np.sqrt(ann)) if ex.std() > 0 else float("nan"),
        "sortino": float(ex.mean() / np.sqrt((down ** 2).mean()) * np.sqrt(ann)) if len(down) else float("nan"),
        "max_drawdown": max_drawdown(eq),
        "n_days": int(len(r)),
    }


def simulate_long_cash(prob: pd.Series, fwd_logret_1: pd.Series, rf_annual_pct: pd.Series,
                       threshold: float = 0.5, cost_bps: float = 5.0, slippage_bps: float = 2.0) -> dict:
    """prob[t] is known at the close of t; fwd_logret_1[t] is the return from close t to close t+1."""
    idx = prob.dropna().index.intersection(fwd_logret_1.dropna().index)
    p, r = prob.loc[idx], np.expm1(fwd_logret_1.loc[idx])
    rf = (rf_annual_pct.reindex(idx).ffill().fillna(0) / 100) / 252
    pos = (p > threshold).astype(float)
    turnover = pos.diff().abs().fillna(pos.iloc[0])
    cost = turnover * (cost_bps + slippage_bps) / 1e4
    strat = pos * r + (1 - pos) * rf - cost
    bh = r
    res = {
        "strategy": performance(strat, rf),
        "buy_and_hold": performance(bh, rf),
        "exposure": float(pos.mean()),
        "turnover_per_year": float(turnover.sum() / len(idx) * 252),
        "n_trades": int((turnover > 0).sum()),
        "cost_bps_round_trip": 2 * (cost_bps + slippage_bps),
    }
    res["daily"] = pd.DataFrame({"position": pos, "strategy": strat, "buy_and_hold": bh, "rf": rf})
    return res


def by_regime(daily: pd.DataFrame, regime: pd.Series) -> dict:
    """Strategy vs buy-and-hold performance split by an ex-ante regime label."""
    out = {}
    reg = regime.reindex(daily.index)
    for name, g in daily.groupby(reg):
        out[str(name)] = {"strategy": performance(g["strategy"], g["rf"]),
                          "buy_and_hold": performance(g["buy_and_hold"], g["rf"]),
                          "exposure": float(g["position"].mean())}
    return out
