"""Experiment framework: model comparison, category ablation, pre-registered selection, final holdout.

Protocol (all decisions before the holdout are made on the development walk-forward only):
  1. `compare`  - every candidate model on the price-only and all-category feature sets (dev).
  2. `ablate`   - price baseline vs price + each category, plus the astrology protocol
                  (calendar-only, astro-only, price+calendar, price+calendar+astro), same folds, same
                  tuning budget. Paired Diebold-Mariano tests vs the price baseline with Holm correction.
  3. `select`   - per task, a category "survives" if its improvement is significant after Holm
                  correction for the task's primary model; the final model is the best dev
                  configuration. Written to selection.json BEFORE the holdout is touched.
  4. `holdout`  - single evaluation on the untouched final period, recorded in holdout_lock.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..backtesting.simulate import by_regime, simulate_long_cash
from ..config import Config
from ..data.store import DataStore
from ..evaluation.metrics import obs_loss, summarize
from ..evaluation.stats import block_bootstrap_ci, diebold_mariano, holm_adjust, kupiec_pof
from ..features.build import columns_for, load_features
from ..logging_utils import get_logger, log_event
from ..targets.labels import build_targets, task_specs
from .walkforward import run_walk_forward

log = get_logger("experiments")

REFERENCE = {"binary": "climatology", "multiclass": "climatology", "regression": "hist_mean",
             "volatility": "rw_vol"}
VOL_REGIME_BINS = [0, 0.12, 0.20, np.inf]   # ex-ante 63-day realised vol (annualised), pre-specified
VOL_REGIME_LABELS = ["low_vol", "mid_vol", "high_vol"]


@dataclass
class Dataset:
    X: pd.DataFrame
    T: pd.DataFrame
    tasks: dict[str, dict]
    context: dict
    meta: dict
    prices: pd.DataFrame
    regimes: pd.Series = field(default=None)


def load_dataset(cfg: Config) -> Dataset:
    store = DataStore(cfg.data_dir)
    X, meta = load_features(cfg, store)
    prices = store.load_prices(cfg.market.ticker)
    T = build_targets(prices["close"], cfg.targets).reindex(X.index)
    ctx = {"returns": np.log(prices["close"]).diff()}
    tasks = {t["task"]: t for t in task_specs(cfg.targets)}
    reg = pd.cut(X["price__rv_63"], VOL_REGIME_BINS, labels=VOL_REGIME_LABELS).astype(str)
    return Dataset(X, T, tasks, ctx, meta, prices, reg)


def feature_sets(cfg: Config) -> dict[str, list[str]]:
    cats = cfg.features.categories
    fs = {"price": ["price"]}
    for c in cats:
        if c != "price":
            fs[f"price+{c}"] = ["price", c]
    if "astro" in cats and "calendar" in cats:
        fs["calendar_only"] = ["calendar"]
        fs["astro_only"] = ["astro"]
        fs["price+calendar+astro"] = ["price", "calendar", "astro"]
    fs["all"] = list(cats)
    return fs


def pred_matrix(df: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "multiclass":
        return df[[c for c in df.columns if c.startswith("p") and c[1:].isdigit()]].to_numpy()
    return df["pred"].to_numpy()


def score(df: pd.DataFrame, kind: str) -> dict:
    y = df["y"].to_numpy()
    base = df["train_mean"].to_numpy() if kind == "regression" else (
        float(df["train_mean"].mean()) if kind == "binary" else None)
    return summarize(kind, y, pred_matrix(df, kind), base)


def paired_test(a: pd.DataFrame, b: pd.DataFrame, kind: str, horizon: int, cfg: Config) -> dict:
    """Compare b against reference a on common dates. Positive mean_diff => b has lower loss."""
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    y = a["y"].to_numpy()
    la, lb = obs_loss(kind, y, pred_matrix(a, kind)), obs_loss(kind, y, pred_matrix(b, kind))
    dm = diebold_mariano(la, lb, horizon)
    d = la - lb
    e = cfg.evaluation
    ci = block_bootstrap_ci(lambda ix: d[ix].mean(), len(d), max(e.bootstrap_block, horizon),
                            e.bootstrap_reps, e.alpha, cfg.project.seed)
    out = {"n_common": len(idx), "loss_ref": float(la.mean()), "loss_model": float(lb.mean()),
           "loss_improvement": dm["mean_diff"], "rel_improvement": float(dm["mean_diff"] / la.mean()),
           "dm_stat": dm["dm_stat"], "p_value": dm["p_value"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]}
    if kind == "binary":
        from sklearn.metrics import roc_auc_score

        pa, pb = pred_matrix(a, kind), pred_matrix(b, kind)

        def auc_diff(ix: np.ndarray) -> float:
            if len(np.unique(y[ix])) < 2:
                return np.nan
            return roc_auc_score(y[ix], pb[ix]) - roc_auc_score(y[ix], pa[ix])
        c2 = block_bootstrap_ci(auc_diff, len(y), max(e.bootstrap_block, horizon), e.bootstrap_reps, e.alpha,
                                cfg.project.seed)
        out.update({"auc_diff": c2["estimate"], "auc_diff_ci_low": c2["ci_low"], "auc_diff_ci_high": c2["ci_high"]})
    return out


def regime_breakdown(a: pd.DataFrame, b: pd.DataFrame, kind: str, regimes: pd.Series) -> dict:
    idx = a.index.intersection(b.index)
    y = a.loc[idx, "y"].to_numpy()
    d = obs_loss(kind, y, pred_matrix(a.loc[idx], kind)) - obs_loss(kind, y, pred_matrix(b.loc[idx], kind))
    s = pd.Series(d, index=idx).groupby(regimes.reindex(idx)).agg(["mean", "size"])
    return {f"impr_{k}": float(v) for k, v in s["mean"].items()} | {f"n_{k}": int(v) for k, v in s["size"].items()}


class Experiment:
    def __init__(self, cfg: Config, run_dir: Path | None = None, ds: Dataset | None = None) -> None:
        self.cfg = cfg
        self.ds = ds or load_dataset(cfg)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = run_dir or cfg.artifacts_dir / "runs" / f"{stamp}_{self.ds.meta['dataset_version']}"
        (self.run_dir / "preds").mkdir(parents=True, exist_ok=True)
        self.fs = feature_sets(cfg)
        self._write_manifest()

    # ---- infrastructure ---------------------------------------------------------------------
    def _write_manifest(self) -> None:
        path = self.run_dir / "manifest.json"
        if path.exists():
            return
        path.write_text(json.dumps({
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset_version": self.ds.meta["dataset_version"],
            "config_fingerprint": self.cfg.fingerprint(),
            "config": self.cfg.model_dump(),
            "feature_sets": self.fs,
            "tasks": list(self.ds.tasks),
        }, indent=2, default=str), encoding="utf-8")

    def _pred_path(self, task: str, model: str, fs: str, period: str) -> Path:
        return self.run_dir / "preds" / f"{period}__{task}__{model}__{fs.replace('+', '-')}.parquet"

    def predictions(self, task: str, model: str, fs: str, period: str) -> pd.DataFrame:
        path = self._pred_path(task, model, fs, period)
        if path.exists():
            return pd.read_parquet(path)
        t = self.ds.tasks[task]
        cols = columns_for(self.ds.X.columns, self.fs[fs])
        df = run_walk_forward(self.ds.X, self.ds.T[t["target"]], t, model, cols, self.cfg, period, self.ds.context)
        df.to_parquet(path)
        return df

    def _run_jobs(self, jobs: list[tuple[str, str, str, str]]) -> None:
        todo = [j for j in dict.fromkeys(jobs) if not self._pred_path(*j).exists()]
        log_event(log, "running jobs", total=len(set(jobs)), todo=len(todo), n_jobs=self.cfg.compute.n_jobs)
        if not todo:
            return
        # Slow jobs first for better load balancing.
        todo.sort(key=lambda j: (j[1] not in ("rf", "lgbm", "xgb", "ar1"), j[0].startswith("regime")), reverse=False)
        Parallel(n_jobs=self.cfg.compute.n_jobs, verbose=0)(delayed(_job)(self, *j) for j in todo)

    def models_for(self, kind: str, ablation: bool = False) -> list[str]:
        m = self.cfg.models
        if ablation:
            return {"binary": m.ablation_classification, "multiclass": m.ablation_classification,
                    "regression": m.ablation_regression, "volatility": m.ablation_volatility}[kind]
        return {"binary": m.classification, "multiclass": m.regime, "regression": m.regression,
                "volatility": m.volatility}[kind]

    # ---- 1. model comparison ----------------------------------------------------------------
    def compare(self, period: str = "dev", feature_set_names: tuple[str, ...] = ("price", "all")) -> pd.DataFrame:
        jobs = [(t, m, fs, period) for t, spec in self.ds.tasks.items() for fs in feature_set_names
                for m in self.models_for(spec["kind"])]
        self._run_jobs(jobs)
        rows = []
        for t, spec in self.ds.tasks.items():
            kind, ref = spec["kind"], REFERENCE[spec["kind"]]
            ref_df = self.predictions(t, ref, "price", period)
            for fs in feature_set_names:
                for m in self.models_for(kind):
                    df = self.predictions(t, m, fs, period)
                    row = {"task": t, "kind": kind, "horizon": spec["horizon"], "feature_set": fs, "model": m,
                           "calibrated": bool(df.attrs.get("calibrated", kind == "binary" and m in self.cfg.models.calibrate)),
                           **{k: v for k, v in score(df, kind).items() if not isinstance(v, list)}}
                    if m != ref:
                        row.update({f"vs_{ref}_{k}": v for k, v in paired_test(ref_df, df, kind, spec["horizon"],
                                                                               self.cfg).items()})
                    rows.append(row)
        res = pd.DataFrame(rows)
        res.to_csv(self.run_dir / f"comparison_{period}.csv", index=False)
        return res

    # ---- 2. ablation -------------------------------------------------------------------------
    def ablate(self, period: str = "dev") -> pd.DataFrame:
        jobs = [(t, m, fs, period) for t, spec in self.ds.tasks.items() for m in self.models_for(spec["kind"], True)
                for fs in self.fs]
        self._run_jobs(jobs)
        rows = []
        for t, spec in self.ds.tasks.items():
            kind = spec["kind"]
            for m in self.models_for(kind, True):
                base = self.predictions(t, m, "price", period)
                for fs, cats in self.fs.items():
                    df = self.predictions(t, m, fs, period)
                    added = [c for c in cats if c != "price"]
                    cov_cols = columns_for(self.ds.X.columns, added)
                    cov = float(self.ds.X.loc[df.index, cov_cols].notna().mean().mean()) if cov_cols else 1.0
                    row = {"task": t, "kind": kind, "horizon": spec["horizon"], "model": m, "feature_set": fs,
                           "n_features": len(columns_for(self.ds.X.columns, cats)),
                           "test_coverage_added": cov,
                           **{k: v for k, v in score(df, kind).items() if not isinstance(v, list)}}
                    if fs != "price":
                        row.update(paired_test(base, df, kind, spec["horizon"], self.cfg))
                        row.update(regime_breakdown(base, df, kind, self.ds.regimes))
                    rows.append(row)
        res = pd.DataFrame(rows)
        single = res["feature_set"].str.match(r"^price\+[a-z_]+$")
        res["holm_p"] = np.nan
        for (_, _), g in res[single].groupby(["task", "model"]):
            res.loc[g.index, "holm_p"] = holm_adjust(g["p_value"].tolist())
        res["significant_improvement"] = (res["holm_p"] < self.cfg.evaluation.alpha) & (res["loss_improvement"] > 0)
        res.to_csv(self.run_dir / f"ablation_{period}.csv", index=False)
        return res

    # ---- 3. pre-registered selection ---------------------------------------------------------
    def select(self, comparison: pd.DataFrame, ablation: pd.DataFrame) -> dict:
        sel = {}
        lossc = {"binary": "log_loss", "multiclass": "log_loss", "regression": "rmse", "volatility": "mse_log"}
        for t, spec in self.ds.tasks.items():
            kind = spec["kind"]
            lc = lossc[kind]
            ab = ablation[ablation["task"] == t]
            base_rows = ab[ab["feature_set"] == "price"]
            primary = base_rows.sort_values(lc).iloc[0]["model"]
            ab_p = ab[(ab["model"] == primary) & ab["significant_improvement"].fillna(False).astype(bool)]
            surviving = sorted(fs.split("+", 1)[1] for fs in ab_p["feature_set"])
            combined = ["price"] + surviving
            self.fs[f"combined_{t}"] = combined
            # Dev score of the combined set is optimistically biased (categories were chosen on dev);
            # it is reported for information only. The holdout gives the unbiased estimate.
            comb = score(self.predictions(t, primary, f"combined_{t}", "dev"), kind)
            cand = comparison[comparison["task"] == t][["model", "feature_set", lc]].copy()
            best = cand.sort_values(lc).iloc[0]
            sel[t] = {"kind": kind, "primary_ablation_model": primary, "surviving_categories": surviving,
                      "combined_feature_set": combined, "combined_dev_loss_biased": float(comb[lc]),
                      "best_dev_model": best["model"], "best_dev_feature_set": best["feature_set"],
                      "best_dev_loss": float(best[lc]), "loss_metric": lc, "reference": REFERENCE[kind]}
        (self.run_dir / "selection.json").write_text(json.dumps({
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rule": "category survives if Holm-adjusted DM p < alpha and mean loss improves, for the task's "
                    "primary ablation model (lowest dev loss on price-only)", "tasks": sel}, indent=2),
            encoding="utf-8")
        return sel

    # ---- 4. final holdout ----------------------------------------------------------------------
    def holdout(self, selection: dict, force: bool = False) -> pd.DataFrame:
        lock = self.cfg.artifacts_dir / "holdout_lock.json"
        history = json.loads(lock.read_text()) if lock.exists() else []
        if history and not force:
            raise RuntimeError(f"holdout already evaluated {len(history)} time(s) (see {lock}); "
                               "pass --force to re-run (this is recorded)")
        jobs, plan = [], []
        for t, s in selection.items():
            comb_name = f"combined_{t}"
            self.fs[comb_name] = s["combined_feature_set"]
            entries = [(s["reference"], "price", "reference"),
                       (s["best_dev_model"], s["best_dev_feature_set"], "selected"),
                       (s["primary_ablation_model"], "price", "primary_price"),
                       (s["primary_ablation_model"], "all", "primary_all"),
                       (s["primary_ablation_model"], comb_name, "primary_combined")]
            if s["kind"] == "volatility":
                entries += [("har", "price", "benchmark"), ("garch", "price", "benchmark")]
            for m, fs, role in entries:
                jobs.append((t, m, fs, "holdout"))
                plan.append((t, m, fs, role))
        self._run_jobs(jobs)
        rows = []
        for t, m, fs, role in dict.fromkeys(plan):
            spec = self.ds.tasks[t]
            df = self.predictions(t, m, fs, "holdout")
            ref = self.predictions(t, selection[t]["reference"], "price", "holdout")
            row = {"task": t, "role": role, "model": m, "feature_set": fs,
                   "categories": "+".join(self.fs[fs]),
                   **{k: v for k, v in score(df, spec["kind"]).items() if not isinstance(v, list)}}
            if role != "reference":
                row.update({f"vs_ref_{k}": v for k, v in paired_test(ref, df, spec["kind"], spec["horizon"],
                                                                     self.cfg).items()})
            rows.append(row)
        res = pd.DataFrame(rows)
        res.to_csv(self.run_dir / "holdout.csv", index=False)
        history.append({"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run_dir": str(self.run_dir),
                        "dataset_version": self.ds.meta["dataset_version"], "config": self.cfg.fingerprint()})
        lock.write_text(json.dumps(history, indent=2), encoding="utf-8")
        return res

    # ---- trading and risk diagnostics -----------------------------------------------------------
    def trading(self, period: str, model_fs: list[tuple[str, str]], task: str = "direction_1") -> dict:
        e = self.cfg.evaluation
        out = {}
        rf = self.ds.X["macro__y3m"] if "macro__y3m" in self.ds.X else pd.Series(0.0, index=self.ds.X.index)
        for m, fs in model_fs:
            df = self.predictions(task, m, fs, period)
            sim = simulate_long_cash(df["pred"], self.ds.T["fwd_ret_1"], rf, 0.5, e.cost_bps_per_turnover,
                                     e.slippage_bps)
            daily = sim.pop("daily")
            sim["by_vol_regime"] = by_regime(daily, self.ds.regimes)
            out[f"{m}|{fs}"] = sim
        (self.run_dir / f"trading_{period}.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
        return out

    def var_backtest(self, period: str, models: list[str], alpha: float = 0.05) -> pd.DataFrame:
        """Normal VaR from log-vol forecasts; Kupiec test on non-overlapping windows."""
        from scipy.stats import norm

        rows = []
        for h in self.cfg.targets.vol_horizons:
            for m in models:
                df = self.predictions(f"volatility_{h}", m, "price", period)
                sig = np.exp(df["pred"]) * np.sqrt(h / 252)
                ret = self.ds.T.loc[df.index, f"fwd_ret_{h}"]
                viol = (ret < norm.ppf(alpha) * sig).to_numpy()[::h]
                rows.append({"horizon": h, "model": m, **kupiec_pof(viol, alpha)})
        res = pd.DataFrame(rows)
        res.to_csv(self.run_dir / f"var_backtest_{period}.csv", index=False)
        return res


def _job(exp: Experiment, task: str, model: str, fs: str, period: str) -> None:
    exp.predictions(task, model, fs, period)
