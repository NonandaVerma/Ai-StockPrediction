"""Feature-matrix assembly across categories, with provenance, coverage and dataset versioning."""

from __future__ import annotations

import json

import pandas as pd

from ..calendar import decision_times
from ..config import Config
from ..data.store import DataStore, frame_hash
from ..logging_utils import get_logger, log_event
from . import astro, behavior, calendar_features, cross_asset, macro, news, price, valuation, weather
from .base import SEP, FeatureContext

log = get_logger("features")

BUILDERS = {
    "price": price.build,
    "calendar": calendar_features.build,
    "macro": macro.build,
    "valuation": valuation.build,
    "cross_asset": cross_asset.build,
    "news": news.build,
    "weather": weather.build,
    "behavior": behavior.build,
    "astro": astro.build,
}


def category_of(col: str) -> str:
    return col.split(SEP, 1)[0]


def columns_for(columns: list[str] | pd.Index, categories: list[str]) -> list[str]:
    return [c for c in columns if category_of(c) in categories]


def build_context(cfg: Config, store: DataStore) -> FeatureContext:
    prices = store.load_prices(cfg.market.ticker)
    ts = decision_times(prices.index, cfg.market.calendar)
    return FeatureContext(cfg=cfg, prices=prices, decision_ts=ts, series=store.load_series())


def build_features(cfg: Config, store: DataStore, categories: list[str] | None = None) -> pd.DataFrame:
    """Build all configured categories, restricted to dates >= market.start_date. Saves Parquet + metadata."""
    ctx = build_context(cfg, store)
    cats = categories or cfg.features.categories
    parts = []
    for cat in cats:
        df = BUILDERS[cat](ctx)
        parts.append(df)
        log_event(log, "category built", category=cat, n_features=df.shape[1])
    feats = pd.concat(parts, axis=1)
    feats = feats[feats.index >= pd.Timestamp(cfg.market.start_date)]
    feats = feats.astype(float)
    feats.index.name = "date"
    version = frame_hash(feats)
    path = store.features_path(cfg.market.id)
    feats.to_parquet(path)
    meta = {
        "dataset_version": version,
        "config_fingerprint": cfg.fingerprint(),
        "market": cfg.market.id,
        "rows": len(feats),
        "first_date": str(feats.index.min().date()),
        "last_date": str(feats.index.max().date()),
        "provenance": {c: ctx.provenance.get(c, {}) for c in feats.columns},
        "coverage": coverage_report(feats).to_dict(orient="index"),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log_event(log, "features saved", path=str(path), version=version, shape=str(feats.shape))
    return feats


def load_features(cfg: Config, store: DataStore) -> tuple[pd.DataFrame, dict]:
    path = store.features_path(cfg.market.id)
    if not path.exists():
        raise FileNotFoundError("feature matrix missing; run `marketpred build-features`")
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return pd.read_parquet(path), meta


def coverage_report(feats: pd.DataFrame) -> pd.DataFrame:
    """Per-feature first valid date and fraction of non-missing values."""
    rows = {}
    for c in feats.columns:
        s = feats[c]
        first = s.first_valid_index()
        rows[c] = {
            "category": category_of(c),
            "first_valid": str(first.date()) if first is not None else None,
            "last_valid": str(s.last_valid_index().date()) if first is not None else None,
            "coverage": float(s.notna().mean()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")
