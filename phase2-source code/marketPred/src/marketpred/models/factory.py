"""Model zoo with a uniform interface: fit(X, y) then predict(X).

* binary      -> predict returns P(y=1) (float array)
* multiclass  -> predict returns (n, k) class probabilities
* regression / volatility -> predict returns point forecasts (same units as y)

All preprocessing (winsorising, imputation, scaling) is fitted inside each model on training data only.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

N_CLASSES = 4  # regime classes


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to quantiles estimated on the training data."""

    def __init__(self, q: float = 0.005) -> None:
        self.q = q

    def fit(self, X: Any, y: Any = None) -> "Winsorizer":
        X = np.asarray(X, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.lo_ = np.nanquantile(X, self.q, axis=0)
            self.hi_ = np.nanquantile(X, 1 - self.q, axis=0)
        return self

    def transform(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        lo = np.where(np.isnan(self.lo_), -np.inf, self.lo_)
        hi = np.where(np.isnan(self.hi_), np.inf, self.hi_)
        return np.clip(X, lo, hi)


def linear_pipeline(estimator: Any, q: float) -> Pipeline:
    return Pipeline([
        ("winsor", Winsorizer(q)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


class Model:
    name = "model"
    kind = "binary"

    def __init__(self, kind: str, params: dict | None = None, horizon: int = 1,
                 context: dict | None = None, seed: int = 42, winsor_q: float = 0.005) -> None:
        self.kind, self.params, self.horizon = kind, dict(params or {}), horizon
        self.context, self.seed, self.winsor_q = context or {}, seed, winsor_q

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Model":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class Climatology(Model):
    """Unconditional training frequency (binary/multiclass) - the no-skill benchmark."""

    name = "climatology"

    def fit(self, X, y):
        if self.kind == "multiclass":
            self.p_ = np.bincount(y.astype(int), minlength=N_CLASSES) / len(y)
        else:
            self.p_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.tile(self.p_, (len(X), 1)) if self.kind == "multiclass" else np.full(len(X), self.p_)


class HistMean(Model):
    """Historical average of the training target (Campbell & Thompson 2008 benchmark)."""

    name = "hist_mean"

    def fit(self, X, y):
        self.m_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m_)


class RandomWalkVol(Model):
    """Log realised volatility over the most recent comparable window (no fitting)."""

    name = "rw_vol"

    def _col(self) -> str:
        return "price__log_rv_5" if self.horizon <= 5 else "price__log_rv_21"

    def fit(self, X, y):
        self.fallback_ = float(np.mean(y))
        return self

    def predict(self, X):
        return X[self._col()].fillna(self.fallback_).to_numpy()


HAR_COLS = ["price__range_1", "price__log_rv_5", "price__log_rv_21", "price__log_rv_63"]


class HAR(Model):
    """HAR-type regression of future log RV on daily/weekly/monthly/quarterly volatility (Corsi 2009).

    The daily component uses the log high-low range because only daily bars are available.
    """

    name = "har"

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        D = X[HAR_COLS].copy()
        D["price__range_1"] = np.log(D["price__range_1"].clip(lower=1e-4))
        return D

    def fit(self, X, y):
        D = self._design(X)
        ok = D.notna().all(axis=1)
        self.m_ = LinearRegression().fit(D[ok], y[ok])
        self.fallback_ = float(np.mean(y))
        return self

    def predict(self, X):
        D = self._design(X)
        ok = D.notna().all(axis=1).to_numpy()
        out = np.full(len(X), self.fallback_)
        out[ok] = self.m_.predict(D[ok])
        return out


class AR1(Model):
    """ARIMA(1,0,0) on daily returns; h-day forecast = sum of k-step-ahead conditional means."""

    name = "ar1"

    def fit(self, X, y):
        from statsmodels.tsa.arima.model import ARIMA

        r = X["price__ret_1"].dropna().to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(r, order=(1, 0, 0), trend="c").fit()
        self.mu_, self.phi_ = float(res.params[0]), float(res.params[1])
        return self

    def predict(self, X):
        h, phi = self.horizon, self.phi_
        dev = X["price__ret_1"].fillna(self.mu_).to_numpy() - self.mu_
        mult = phi * (1 - phi ** h) / (1 - phi) if abs(phi) < 1 else 0.0
        return h * self.mu_ + dev * mult


class Garch(Model):
    """GARCH(1,1) with Student-t errors on daily returns; parameters estimated on the training window,
    then held fixed while filtering forward, so each forecast uses returns up to its origin only."""

    name = "garch"

    def fit(self, X, y):
        from arch import arch_model

        self.r_ = self.context["returns"].dropna() * 100
        train_r = self.r_[self.r_.index <= X.index.max()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.params_ = arch_model(train_r, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(
                disp="off").params
        self.fallback_ = float(np.mean(y))
        return self

    def predict(self, X):
        from arch import arch_model

        r = self.r_[self.r_.index <= X.index.max()]
        fixed = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fix(self.params_)
        fc = fixed.forecast(horizon=self.horizon, start=X.index.min(), reindex=False)
        var_sum = fc.variance.sum(axis=1) / 1e4  # back to decimal returns
        log_rv = 0.5 * np.log(252.0 / self.horizon * var_sum)
        return log_rv.reindex(X.index).fillna(self.fallback_).to_numpy()


class SkPipelineModel(Model):
    """Wrapper for sklearn-compatible estimators (linear models and tree ensembles)."""

    def __init__(self, name: str, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.name = name

    def _estimator(self) -> Any:
        p, s, k = self.params, self.seed, self.kind
        clf = k in ("binary", "multiclass")
        if self.name == "logit":
            return linear_pipeline(LogisticRegression(random_state=s, **p), self.winsor_q)
        if self.name == "ridge":
            return linear_pipeline(Ridge(random_state=s, **p), self.winsor_q)
        if self.name == "enet":
            return linear_pipeline(ElasticNet(random_state=s, **p), self.winsor_q)
        if self.name == "lgbm":
            import lightgbm as lgb

            cls = lgb.LGBMClassifier if clf else lgb.LGBMRegressor
            return cls(random_state=s, **p)
        if self.name == "xgb":
            import xgboost as xgb

            cls = xgb.XGBClassifier if clf else xgb.XGBRegressor
            return cls(random_state=s, tree_method="hist", **p)
        if self.name == "rf":
            cls = RandomForestClassifier if clf else RandomForestRegressor
            return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                             ("model", cls(random_state=s, **p))])
        if self.name == "catboost":
            import catboost

            cls = catboost.CatBoostClassifier if clf else catboost.CatBoostRegressor
            return cls(random_seed=s, verbose=False, **p)
        raise ValueError(f"unknown model {self.name}")

    def fit(self, X, y):
        self.cols_ = [c for c in X.columns if X[c].notna().any()]  # drop columns with no training data
        self.est_ = self._estimator()
        yy = y.astype(int) if self.kind in ("binary", "multiclass") else y
        if self.kind == "multiclass":
            self.classes_ = np.unique(yy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.est_.fit(X[self.cols_], yy)
        return self

    def predict(self, X):
        Xc = X[self.cols_]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.kind == "binary":
                return self.est_.predict_proba(Xc)[:, 1]
            if self.kind == "multiclass":
                p = self.est_.predict_proba(Xc)
                full = np.zeros((len(X), N_CLASSES))
                full[:, self.classes_] = p
                return full
            return self.est_.predict(Xc)


SPECIAL = {"climatology": Climatology, "hist_mean": HistMean, "rw_vol": RandomWalkVol, "har": HAR,
           "ar1": AR1, "garch": Garch}
FEATURE_FREE = {"climatology", "hist_mean"}            # do not use features
FIXED_INPUT = {"rw_vol", "har", "ar1", "garch"}        # use a fixed set of price inputs


def make_model(name: str, kind: str, params: dict | None = None, horizon: int = 1,
               context: dict | None = None, seed: int = 42, winsor_q: float = 0.005) -> Model:
    if name in SPECIAL:
        return SPECIAL[name](kind, params, horizon, context, seed, winsor_q)
    return SkPipelineModel(name, kind, params, horizon, context, seed, winsor_q)


def available_models() -> dict[str, bool]:
    out = {}
    for mod in ("lightgbm", "xgboost", "catboost"):
        try:
            __import__(mod)
            out[mod] = True
        except ImportError:
            out[mod] = False
    return out
