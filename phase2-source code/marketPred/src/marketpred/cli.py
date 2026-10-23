"""Command-line entry point: `python -m marketpred <command>` (or the `marketpred` script).

Typical workflow:
    marketpred ingest            # download all keyless sources (raw snapshots + processed series)
    marketpred build-features    # point-in-time feature matrix
    marketpred evaluate          # dev walk-forward: model comparison + ablation + selection
    marketpred holdout           # one-time final test on the untouched holdout period
    marketpred report            # write docs/RESULTS.md from the latest run
    marketpred predict           # structured forecast for the latest session
    marketpred monitor           # feature drift report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .logging_utils import get_logger, setup_logging

log = get_logger("cli")


def _latest_run(cfg) -> Path:
    runs = sorted((cfg.artifacts_dir / "runs").glob("*/manifest.json"))
    if not runs:
        raise SystemExit("no runs found; run `marketpred evaluate` first")
    return runs[-1].parent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="marketpred")
    p.add_argument("--config", default=None, help="YAML file deep-merged over configs/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest", help="download and store all enabled sources")
    s.add_argument("--sources", nargs="*", default=None)
    sub.add_parser("build-features", help="build the point-in-time feature matrix")
    s = sub.add_parser("evaluate", help="dev walk-forward: comparison, ablation, selection")
    s.add_argument("--run-dir", default=None, help="resume/extend an existing run directory")
    s.add_argument("--quick", action="store_true", help="smoke mode: 2 tasks, fewer models")
    s = sub.add_parser("holdout", help="final evaluation on the untouched holdout period")
    s.add_argument("--run-dir", default=None)
    s.add_argument("--force", action="store_true", help="re-run even if the holdout was already used")
    s = sub.add_parser("report", help="write docs/RESULTS.md from a run")
    s.add_argument("--run-dir", default=None)
    s = sub.add_parser("predict", help="forecast record for the latest session")
    s.add_argument("--as-of", default=None)
    sub.add_parser("monitor", help="feature drift report")
    args = p.parse_args(argv)

    overrides = None
    if getattr(args, "quick", False):
        overrides = {"targets": {"horizons": [1, 5], "vol_horizons": [5], "downside": {5: -0.02}},
                     "models": {"classification": ["climatology", "logit", "lgbm"],
                                "regression": ["hist_mean", "ridge", "lgbm"],
                                "volatility": ["rw_vol", "har", "lgbm"], "regime": ["climatology", "logit"]}}
    cfg = load_config(args.config, overrides)
    setup_logging(cfg.artifacts_dir / "logs")

    if args.cmd == "ingest":
        from .data.ingest import run_ingest

        print(json.dumps(run_ingest(cfg, args.sources), indent=2))
    elif args.cmd == "build-features":
        from .data.store import DataStore
        from .features.build import build_features

        f = build_features(cfg, DataStore(cfg.data_dir))
        print(f"features: {f.shape[0]} rows x {f.shape[1]} columns, {f.index.min().date()}..{f.index.max().date()}")
    elif args.cmd == "evaluate":
        from .training.experiments import Experiment

        run_dir = Path(args.run_dir) if args.run_dir else None
        if args.quick and run_dir is None:  # keep smoke runs out of artifacts/runs
            from datetime import datetime, timezone

            run_dir = cfg.artifacts_dir / "quick_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        exp = Experiment(cfg, run_dir)
        comp = exp.compare("dev")
        abl = exp.ablate("dev")
        sel = exp.select(comp, abl)
        exp.trading("dev", [(m, fs) for m in cfg.models.classification for fs in ("price", "all")])
        exp.var_backtest("dev", [m for m in ("rw_vol", "har", "garch", "lgbm") if m in cfg.models.volatility])
        print(f"run directory: {exp.run_dir}")
        print(json.dumps({t: {k: s[k] for k in ("best_dev_model", "best_dev_feature_set", "surviving_categories")}
                          for t, s in sel.items()}, indent=2))
    elif args.cmd == "holdout":
        from .training.experiments import Experiment

        run_dir = Path(args.run_dir) if args.run_dir else _latest_run(cfg)
        sel = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))["tasks"]
        exp = Experiment(cfg, run_dir)
        res = exp.holdout(sel, force=args.force)
        d1 = sel.get("direction_1")
        pairs = [(m, "price") for m in cfg.models.classification]
        if d1:
            pairs.append((d1["best_dev_model"], d1["best_dev_feature_set"]))
        exp.trading("holdout", list(dict.fromkeys(pairs)))
        exp.var_backtest("holdout", [m for m in ("rw_vol", "har", "garch", "lgbm") if m in cfg.models.volatility])
        print(res[["task", "role", "model", "feature_set"]].to_string())
    elif args.cmd == "report":
        from .evaluation.report import write_report

        run_dir = Path(args.run_dir) if args.run_dir else _latest_run(cfg)
        print(write_report(cfg, run_dir))
    elif args.cmd == "predict":
        from .prediction.predict import make_forecast

        rec = make_forecast(cfg, as_of=args.as_of)
        print(json.dumps({k: rec[k] for k in ("asset", "prediction_timestamp_utc", "horizons_trading_days",
                                              "regime_probabilities", "data_quality_warnings")},
                         indent=2, default=float))
    elif args.cmd == "monitor":
        from .data.store import DataStore
        from .features.build import load_features
        from .monitoring.drift import drift_report

        X, _ = load_features(cfg, DataStore(cfg.data_dir))
        rep = drift_report(X)
        out = cfg.artifacts_dir / "monitoring"
        out.mkdir(parents=True, exist_ok=True)
        rep.to_csv(out / "drift_latest.csv", index=False)
        print(rep.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
