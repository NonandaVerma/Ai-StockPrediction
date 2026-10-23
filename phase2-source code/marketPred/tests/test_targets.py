"""Target definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from marketpred.targets.labels import build_targets, forward_sum


def test_forward_sum_definition():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    fs = forward_sum(x, 2)
    assert fs.tolist()[:3] == [5.0, 7.0, 9.0]  # x[i+1] + x[i+2]
    assert fs.iloc[3:].isna().all()


def test_targets_match_manual_computation(prices, cfg):
    T = build_targets(prices["close"], cfg.targets)
    c = prices["close"]
    i = 100
    for h in (1, 5, 20):
        assert np.isclose(T[f"fwd_ret_{h}"].iloc[i], np.log(c.iloc[i + h] / c.iloc[i]))
        assert T[f"dir_{h}"].iloc[i] == float(np.log(c.iloc[i + h] / c.iloc[i]) > 0)
        r = np.log(c).diff().iloc[i + 1:i + h + 1]
        assert np.isclose(T[f"fwd_rv_{h}"].iloc[i], np.sqrt(252 / h * (r ** 2).sum()))
        assert T[f"label_end_{h}"].iloc[i] == prices.index[i + h]
    assert T["down_5"].iloc[i] == float(T["fwd_ret_5"].iloc[i] < -0.02)
    assert T["large_move"].iloc[i] == float(abs(T["fwd_ret_1"].iloc[i]) > 0.02)


def test_targets_nan_when_window_incomplete(prices, cfg):
    T = build_targets(prices["close"], cfg.targets)
    assert T["fwd_ret_20"].iloc[-20:].isna().all() and T["dir_20"].iloc[-20:].isna().all()
    assert T["fwd_ret_20"].iloc[-21:-20].notna().all()
    assert T["regime"].iloc[-20:].isna().all()


def test_regime_definition(prices, cfg):
    T = build_targets(prices["close"], cfg.targets).dropna(subset=["regime"])
    r = cfg.targets.regime
    hv = T["fwd_rv_20"] > r["high_vol_annualized"]
    assert (T.loc[hv, "regime"] == 3).all()
    assert (T.loc[~hv & (T["fwd_ret_20"] > r["bull_return"]), "regime"] == 0).all()
    assert (T.loc[~hv & (T["fwd_ret_20"] < r["bear_return"]), "regime"] == 1).all()


def test_targets_do_not_depend_on_past_changes(prices, cfg):
    """Changing prices strictly before t+1 (except C_t itself) must not alter forward labels at t."""
    T1 = build_targets(prices["close"], cfg.targets)
    p2 = prices["close"].copy()
    p2.iloc[:100] *= 1.5
    T2 = build_targets(p2, cfg.targets)
    assert np.allclose(T1["fwd_ret_5"].iloc[100:-5], T2["fwd_ret_5"].iloc[100:-5])
