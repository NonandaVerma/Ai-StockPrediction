"""Model registry for the forecast engine.

Every model produces two numbers per row: P(window return > 0) and an expected log return.
Families:
  baseline  naive no-change and historical-mean forecasts
  market    conventional market model (returns, volatility, volume, technical indicators)
  astro     astrology-only (Model A: L1-regularised statistical model; plus LightGBM)
  hybrid    market + astrology features (Model C)
  geopolitics  market + geopolitical-risk (GPR) features
  weather      market + India weather (NASA POWER) features, NSE/BSE listings only
Algorithms: naive_zero | naive_mean | linear | lgbm | xgb | ensemble (average of linear and LightGBM).
Training rows, folds and windows are decided by src/training.py, never here.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .repository import stable_hash
from .validation import walk_forward_splits

warnings.filterwarnings("ignore", message=".*feature names.*")
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
CODE_VERSION = "fc1"   # bump when model code changes so stored versions stay distinguishable

try:  # optional dependency
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str                 # baseline | market | astro | hybrid
    groups: tuple               # feature groups: "market", "astro"
    algo: str                   # naive_zero | naive_mean | linear | lgbm | xgb | ensemble
    label: str
    params: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def version(self) -> str:
        return f"{CODE_VERSION}-{stable_hash({'algo': self.algo, 'groups': self.groups, **self.params})}"


LINEAR = {"C": 0.05, "penalty": "l1", "ridge_alphas": [1e1, 1e2, 1e3, 1e4, 1e5]}
GBM = {"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 80,
       "subsample": 0.8, "colsample_bytree": 0.6, "reg_lambda": 5.0}
ENSEMBLE = {"linear": LINEAR, "lgbm": GBM}

# Small, hand-picked hyperparameter grids used when TrainSettings.tune=True (opt-in: see
# FittedModel.fit). GBM/XGB_DEFAULT above/below are always candidate 0, so tuning can never do
# worse than today's fixed defaults, only find something better via nested walk-forward CV.
LGBM_TUNE_GRID = [
    GBM,                                                                                    # current default
    {**GBM, "num_leaves": 7, "learning_rate": 0.05, "min_child_samples": 120, "reg_lambda": 10.0},   # more regularised
    {**GBM, "num_leaves": 31, "learning_rate": 0.1, "min_child_samples": 50, "reg_lambda": 2.0},     # more aggressive
    {**GBM, "num_leaves": 7, "learning_rate": 0.02, "min_child_samples": 150, "reg_lambda": 15.0},   # very conservative
]
XGB_TUNE_GRID = [
    {**GBM, "max_depth": 4},                                                                 # current default
    {**GBM, "max_depth": 3, "learning_rate": 0.05, "reg_lambda": 10.0},                      # more regularised
    {**GBM, "max_depth": 6, "learning_rate": 0.1, "reg_lambda": 2.0},                        # more aggressive
    {**GBM, "max_depth": 3, "learning_rate": 0.02, "reg_lambda": 15.0},                      # very conservative
]
MIN_ROWS_PER_TUNE_FOLD = 100   # too little data for a meaningful inner fold: skip tuning, use the default

SPECS = [
    ModelSpec("naive_zero", "baseline", (), "naive_zero", "Naive: no change"),
    ModelSpec("naive_mean", "baseline", (), "naive_mean", "Naive: historical average"),
    ModelSpec("market_linear", "market", ("market",), "linear", "Market-only (regularised linear)", LINEAR),
    ModelSpec("market_lgbm", "market", ("market",), "lgbm", "Market-only (LightGBM)", GBM),
    ModelSpec("astro_stat", "astro", ("astro",), "linear", "Astro-only statistical (Model A, L1)", LINEAR),
    ModelSpec("astro_lgbm", "astro", ("astro",), "lgbm", "Astro-only (LightGBM)", GBM),
    ModelSpec("hybrid_linear", "hybrid", ("market", "astro"), "linear", "Hybrid market+astro (linear)", LINEAR),
    ModelSpec("hybrid_lgbm", "hybrid", ("market", "astro"), "lgbm", "Hybrid market+astro (LightGBM)", GBM),
    # Average of the linear and LightGBM models on the same inputs: usually steadier than either alone.
    ModelSpec("market_ensemble", "market", ("market",), "ensemble", "Market-only (linear + LightGBM average)", ENSEMBLE),
    ModelSpec("hybrid_ensemble", "hybrid", ("market", "astro"), "ensemble",
              "Hybrid market+astro (linear + LightGBM average)", ENSEMBLE),
    # Geopolitical risk (GPR) and India weather (NASA POWER), lagged to publication: src/risk_external.py
    ModelSpec("market_geo_linear", "geopolitics", ("market", "geopolitics"), "linear",
              "Market + geopolitics (linear)", LINEAR),
    ModelSpec("market_geo_lgbm", "geopolitics", ("market", "geopolitics"), "lgbm",
              "Market + geopolitics (LightGBM)", GBM),
    ModelSpec("market_weather_linear", "weather", ("market", "weather"), "linear",
              "Market + India weather (linear)", LINEAR),
    ModelSpec("market_weather_lgbm", "weather", ("market", "weather"), "lgbm",
              "Market + India weather (LightGBM)", GBM),
]
# One "market + category" pair of models per catalog category (src/catalog_features.py); categories
# 12 and 14 are the geopolitics / weather models above.
from .catalog_features import CATEGORIES as _CATS
for _key, _title in _CATS.items():
    if _key in ("geopolitics", "weather"):
        continue
    for _algo, _p, _tag in (("linear", LINEAR, "linear"), ("lgbm", GBM, "LightGBM")):
        SPECS.append(ModelSpec(f"market_{_key}_{_algo}", "category", ("market", _key), _algo,
                               f"Market + {_title} ({_tag})", _p))
# Kitchen sink: every available category at once. Expected to overfit; shown as a warning sign.
SPECS.append(ModelSpec("all_categories_linear", "all_categories", ("market", "all_categories"), "linear",
                       "Market + all available categories (linear, L1)", LINEAR))
SPECS.append(ModelSpec("all_categories_lgbm", "all_categories", ("market", "all_categories"), "lgbm",
                       "Market + all available categories (LightGBM)", GBM))
if HAS_XGB:
    SPECS.append(ModelSpec("hybrid_xgb", "hybrid", ("market", "astro"), "xgb", "Hybrid (XGBoost)", GBM))

# ---------------------------------------------------------------- custom models (any categories x algorithm)
# Data categories a user can combine: the dashboard market set, the astro set, and every catalog category.
SELECTABLE_GROUPS = {"market": "Market (dashboard set)", "astro": "Astrology (all sky features)", **_CATS}
ALGORITHMS = {"linear": ("Linear", LINEAR), "lgbm": ("LightGBM", GBM), "ensemble": ("Linear + LightGBM", ENSEMBLE)}
CUSTOM_PREFIX = "custom:"


def custom_model_name(groups, algo: str) -> str:
    """Model name for a user-chosen combination, e.g. 'custom:ensemble:astro+weather'."""
    if algo not in ALGORITHMS:
        raise ValueError(f"unknown algorithm '{algo}'")
    groups = [g for g in SELECTABLE_GROUPS if g in set(groups)]          # canonical order, no duplicates
    if not groups:
        raise ValueError("choose at least one data category")
    return f"{CUSTOM_PREFIX}{algo}:{'+'.join(groups)}"


def _custom_spec(name: str) -> ModelSpec:
    algo, _, groups = name.removeprefix(CUSTOM_PREFIX).partition(":")
    groups = tuple(groups.split("+"))
    if algo not in ALGORITHMS or not groups or any(g not in SELECTABLE_GROUPS for g in groups):
        raise KeyError(name)
    tag, params = ALGORITHMS[algo]
    label = " + ".join(SELECTABLE_GROUPS[g] for g in groups)
    return ModelSpec(name, "custom", groups, algo, f"{label} ({tag})", params)


class _SpecRegistry(dict):
    """Fixed models by name; 'custom:<algo>:<group+group>' names are built on demand (never stored, so
    model lists only show the fixed models)."""

    def __missing__(self, name):
        if isinstance(name, str) and name.startswith(CUSTOM_PREFIX):
            return _custom_spec(name)
        raise KeyError(name)


SPEC_BY_NAME = _SpecRegistry({s.name: s for s in SPECS})
DEFAULT_MODELS = ["naive_zero", "naive_mean", "market_linear", "market_ensemble", "astro_stat", "astro_lgbm",
                  "hybrid_lgbm", "market_geo_linear", "market_weather_linear"]
# Model Comparison / Backtesting default: every category's linear model (Home keeps the short list).
COMPARISON_MODELS = list(dict.fromkeys(
    DEFAULT_MODELS + [s.name for s in SPECS if s.family in ("category", "geopolitics", "weather", "all_categories")
                      and s.algo == "linear"]))


def _lgbm_kw(p: dict, seed: int) -> dict:
    return dict(p, subsample_freq=1, random_state=seed, verbose=-1, n_jobs=-1)


def _xgb_kw(p: dict, seed: int) -> dict:
    return dict(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p.get("max_depth", 4),
                subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
                reg_lambda=p["reg_lambda"], random_state=seed, n_jobs=-1)


_ALGO = {"lgbm": (LGBMClassifier, _lgbm_kw, LGBM_TUNE_GRID)}
if HAS_XGB:
    _ALGO["xgb"] = (XGBClassifier, _xgb_kw, XGB_TUNE_GRID)


EARLY_STOP_ROUNDS = 20
MIN_ROWS_FOR_EARLY_STOP = 150   # too little data for a reliable internal validation tail: skip it


def _fit_boosted(algo: str, Model, kw: dict, X: np.ndarray, y: np.ndarray, embargo: int, is_classifier: bool):
    """Fit a LightGBM/XGBoost estimator with early stopping on an internal, time-ordered
    validation tail (the last 15% of rows in this fold, after an embargo gap so no target window
    overlaps) instead of always training the fixed n_estimators - a standard, cheap control against
    overfitting. Falls back to a plain fixed-round fit whenever there isn't enough data for a
    reliable internal split, so this can never do worse than the previous behavior."""
    n = len(y)
    val_n = max(30, int(0.15 * n))
    tr_end = n - val_n - embargo
    ok = n >= MIN_ROWS_FOR_EARLY_STOP and tr_end >= MIN_ROWS_PER_TUNE_FOLD
    if ok and is_classifier:
        ok = len(np.unique(y[:tr_end])) >= 2 and len(np.unique(y[n - val_n:])) >= 2
    if not ok:
        m = Model(**kw)
        m.fit(X, y)
        return m
    kw = {**kw, "n_estimators": 500}       # upper cap; early stopping finds the real count
    Xtr, ytr, Xval, yval = X[:tr_end], y[:tr_end], X[n - val_n:], y[n - val_n:]
    try:
        if algo == "lgbm":
            import lightgbm as lgb
            m = Model(**kw)
            m.fit(Xtr, ytr, eval_set=[(Xval, yval)], callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False)])
        else:
            m = Model(**kw, early_stopping_rounds=EARLY_STOP_ROUNDS)
            m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        return m
    except Exception:
        m = Model(**{**kw, "n_estimators": 200})
        m.fit(X, y)
        return m


def _best_params(algo: str, X: np.ndarray, y: np.ndarray, embargo: int, seed: int) -> dict:
    """Nested walk-forward CV over a small candidate grid: 2 inner folds, scored by mean
    out-of-fold accuracy. Falls back to the current fixed default (grid[0]) when there isn't
    enough data for a meaningful inner fold, so tuning can never do worse than today's behavior."""
    Classifier, build_kw, grid = _ALGO[algo]
    splits = [(np.asarray(tr), np.asarray(te)) for tr, te in walk_forward_splits(len(y), 2, embargo)
             if len(tr) >= MIN_ROWS_PER_TUNE_FOLD and len(te) and len(np.unique(y[tr])) >= 2]
    if not splits:
        return grid[0]
    scores = []
    for params in grid:
        accs = []
        for tr, te in splits:
            if len(np.unique(y[te])) < 1:
                continue
            m = Classifier(**build_kw(params, seed))
            m.fit(X[tr], y[tr])
            accs.append(accuracy_score(y[te], m.predict_proba(X[te])[:, 1] > 0.5))
        scores.append(float(np.mean(accs)) if accs else -1.0)
    return grid[int(np.argmax(scores))]


class FittedModel:
    def __init__(self, spec: ModelSpec, seed: int = 42, tune: bool = False):
        self.spec, self.seed, self.tune = spec, seed, tune

    def fit(self, X: np.ndarray, y_ret: np.ndarray, clf_rows: np.ndarray | None = None, embargo: int = 1):
        """`clf_rows`: optional boolean mask of the rows the up/down classifier learns from (the return
        regressor always uses every row). Used to leave out near-zero moves, whose sign is mostly noise.
        `embargo`: gap+horizon rows, needed to run a correct inner walk-forward split when `self.tune`."""
        y_up = (y_ret > 0).astype(int)
        self.up_rate, self.mean_ret = float(y_up.mean()), float(y_ret.mean())
        a, p = self.spec.algo, self.spec.params
        if a in ("naive_zero", "naive_mean"):
            return self
        if a == "ensemble":
            self.members = [FittedModel(ModelSpec(f"{self.spec.name}:{k}", self.spec.family, self.spec.groups, k,
                                                  "", p[k]), self.seed, self.tune).fit(X, y_ret, clf_rows, embargo)
                            for k in ("linear", "lgbm")]
            return self
        if clf_rows is not None and clf_rows.sum() >= 50:
            Xc, yc = X[clf_rows], y_up[clf_rows]
        else:
            Xc, yc = X, y_up
        if a == "linear":
            self.clf = make_pipeline(StandardScaler(), LogisticRegression(
                C=p["C"], l1_ratio=1.0 if p["penalty"] == "l1" else 0.0, solver="saga", max_iter=3000, random_state=self.seed))
            # penalty chosen by chronological (not shuffled) inner CV
            self.reg = make_pipeline(StandardScaler(), RidgeCV(alphas=p["ridge_alphas"], cv=TimeSeriesSplit(3)))
        elif a in ("lgbm", "xgb"):
            Classifier, build_kw, grid = _ALGO[a]
            Regressor = LGBMRegressor if a == "lgbm" else XGBRegressor
            params = _best_params(a, Xc, yc, embargo, self.seed) if self.tune and len(np.unique(yc)) >= 2 else grid[0]
            self.tuned_params = params if self.tune else None
            kw = build_kw(params, self.seed)
            if len(np.unique(yc)) >= 2:
                self.clf = _fit_boosted(a, Classifier, kw, Xc, yc, embargo, True)
            else:
                self.clf = None
            self.reg = _fit_boosted(a, Regressor, kw, X, y_ret, embargo, False)
            return self
        if len(np.unique(yc)) < 2:                   # degenerate window: fall back to base rate
            self.clf = None
        else:
            self.clf.fit(Xc, yc)
        self.reg.fit(X, y_ret)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(X)
        a = self.spec.algo
        if a == "naive_zero":
            return np.full(n, 0.5), np.zeros(n)
        if a == "naive_mean":
            return np.full(n, self.up_rate), np.full(n, self.mean_ret)
        if a == "ensemble":                          # plain average of the linear and LightGBM members
            out = [m.predict(X) for m in self.members]
            return np.mean([o[0] for o in out], axis=0), np.mean([o[1] for o in out], axis=0)
        p = self.clf.predict_proba(X)[:, 1] if self.clf is not None else np.full(n, self.up_rate)
        return p, self.reg.predict(X)

    def selected_features(self, names: list[str]) -> list[str]:
        """Features kept by the L1 penalty (linear models only)."""
        if self.spec.algo == "ensemble":
            return self.members[0].selected_features(names)
        if self.spec.algo != "linear" or self.clf is None:
            return []
        coef = self.clf[-1].coef_[0]
        return [n for n, c in sorted(zip(names, coef), key=lambda t: -abs(t[1])) if abs(c) > 1e-8]
