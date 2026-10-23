"""Leakage tests for the validation machinery: purging, embargo, holdout isolation, train-only fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketpred.models.factory import Winsorizer, make_model
from marketpred.training.walkforward import calibration_split, run_walk_forward, walk_forward_folds


@pytest.mark.parametrize("h", [1, 5, 20])
def test_purged_folds_have_no_label_overlap(sessions, h):
    folds = walk_forward_folds(sessions, "2018-01-01", "2020-01-01", h, 63, 5, 300, label_limit="2020-01-01")
    assert folds
    lim = sessions.searchsorted(pd.Timestamp("2020-01-01"))
    for f in folds:
        # label of a train row i covers i+1..i+h; must end before the first test row (minus embargo)
        assert f.train.max() + h < f.test.min() - 5
        assert f.train.min() == 0 and (np.diff(f.test) == 1).all()
        assert (f.test + h < lim).all()  # dev labels never reach the holdout


def test_folds_cover_test_period_in_order(sessions):
    folds = walk_forward_folds(sessions, "2018-01-01", None, 1, 63, 5, 300)
    tests = np.concatenate([f.test for f in folds])
    assert (np.diff(tests) == 1).all()
    assert sessions[tests[0]] >= pd.Timestamp("2018-01-01") and tests[-1] == len(sessions) - 1


def test_calibration_split_is_purged():
    tr = np.arange(1000)
    fit, cal = calibration_split(tr, 0.2, 20, 5)
    assert fit.max() + 20 < cal.min() - 5 + 1 and len(cal) == 200


def test_winsorizer_uses_training_quantiles_only():
    train = np.arange(100, dtype=float).reshape(-1, 1)
    w = Winsorizer(0.05).fit(train)
    out = w.transform(np.array([[1e9], [-1e9]]))
    assert out[0, 0] <= 99 and out[1, 0] >= 0


def _toy(sessions, leak: bool):
    rng = np.random.default_rng(3)
    n = len(sessions)
    y = pd.Series((rng.random(n) > 0.5).astype(float), index=sessions)
    X = pd.DataFrame({"price__noise": rng.normal(size=n), "price__ret_1": rng.normal(size=n) * 0.01},
                     index=sessions)
    if leak:
        X["price__noise"] = y + rng.normal(scale=0.01, size=n)  # contemporaneous label leak
    return X, y


def test_walk_forward_detects_planted_leak_and_finds_no_skill_in_noise(sessions, cfg):
    """Sanity check of the harness: a planted leak is detected (AUC~1); pure noise gives AUC~0.5."""
    from sklearn.metrics import roc_auc_score

    c = cfg.model_copy(deep=True)
    c.validation.dev_start, c.validation.holdout_start, c.validation.min_train_obs = "2018-01-01", "2020-06-01", 300
    task = {"task": "t", "kind": "binary", "target": "y", "horizon": 1}
    X, y = _toy(sessions, leak=False)
    r = run_walk_forward(X, y, task, "logit", ["price__noise"], c, "dev")
    assert abs(roc_auc_score(r["y"], r["pred"]) - 0.5) < 0.06
    X, y = _toy(sessions, leak=True)
    r = run_walk_forward(X, y, task, "logit", ["price__noise"], c, "dev")
    assert roc_auc_score(r["y"], r["pred"]) > 0.95


def test_walk_forward_predictions_unaffected_by_future_data(sessions, cfg):
    """Corrupting features/labels after a fold's test block must not change that fold's predictions."""
    c = cfg.model_copy(deep=True)
    c.validation.dev_start, c.validation.holdout_start, c.validation.min_train_obs = "2018-01-01", "2020-06-01", 300
    c.validation.retrain_every_days = 63
    task = {"task": "t", "kind": "binary", "target": "y", "horizon": 5}
    X, y = _toy(sessions, leak=False)
    base = run_walk_forward(X, y, task, "lgbm", ["price__noise", "price__ret_1"], c, "dev")
    first = base[base["fold"] == 0]
    cut = X.index.get_loc(first.index[-1])
    X2, y2 = X.copy(), y.copy()
    X2.iloc[cut + 1:] = 999.0
    y2.iloc[cut + 1:] = 1.0 - y2.iloc[cut + 1:]
    pert = run_walk_forward(X2, y2, task, "lgbm", ["price__noise", "price__ret_1"], c, "dev")
    np.testing.assert_allclose(first["pred"].to_numpy(), pert.loc[first.index, "pred"].to_numpy())


def test_garch_forecasts_use_only_past_returns(sessions):
    rng = np.random.default_rng(5)
    r = pd.Series(rng.standard_t(5, len(sessions)) * 0.01, index=sessions)
    X = pd.DataFrame({"price__ret_1": r}, index=sessions)
    y = pd.Series(np.log(0.15), index=sessions)
    tr, te = sessions[:800], sessions[800:900]
    m1 = make_model("garch", "volatility", horizon=5, context={"returns": r}).fit(X.loc[tr], y.loc[tr])
    p1 = m1.predict(X.loc[te])
    r2 = r.copy()
    r2.iloc[900:] *= 10  # future returns changed
    m2 = make_model("garch", "volatility", horizon=5, context={"returns": r2}).fit(X.loc[tr], y.loc[tr])
    np.testing.assert_allclose(p1, m2.predict(X.loc[te]))
