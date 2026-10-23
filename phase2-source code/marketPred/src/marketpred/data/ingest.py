"""Ingestion orchestrator: fetch -> raw snapshot -> transform -> validate -> processed Parquet.

A failing or unauthorised source is logged and skipped; nothing is fabricated in its place.
"""

from __future__ import annotations

import os

import pandas as pd

from ..calendar import decision_times, trading_days
from ..config import Config
from ..http import HttpClient
from ..logging_utils import get_logger, log_event
from .connectors import CONNECTORS
from .connectors.yahoo import fetch_history
from .store import DataStore
from .validation import check_calendar_alignment, validate_prices, validate_series

log = get_logger("ingest")


def make_client(cfg: Config) -> HttpClient:
    h = cfg.http
    return HttpClient(cfg.data_dir / "cache", timeout_s=h.timeout_s, max_retries=h.max_retries,
                      backoff_factor=h.backoff_factor, min_interval_s=h.min_interval_s,
                      cache_ttl_hours=h.cache_ttl_hours)


def ingest_prices(cfg: Config, store: DataStore, client: HttpClient) -> pd.DataFrame:
    """Fetch, calendar-check and store OHLCV for the target market."""
    m = cfg.market
    df = fetch_history(client, m.ticker, "1900-01-01")
    store.save_raw("yahoo_prices", m.ticker, df)
    df = df[df.index >= pd.Timestamp(m.start_date) - pd.Timedelta(days=400)]  # lookback buffer
    sessions = trading_days(m.calendar, str(df.index.min().date()), str(df.index.max().date()))
    cal = check_calendar_alignment(df.index, sessions)
    if cal["not_sessions"]:
        log.warning(f"{m.ticker}: dropping {len(cal['not_sessions'])} non-session dates "
                    f"e.g. {cal['not_sessions'][:3]}")
        df = df.drop(index=cal["not_sessions"])
    if cal["missing_sessions"]:
        log.warning(f"{m.ticker}: {len(cal['missing_sessions'])} exchange sessions missing from provider "
                    f"e.g. {[str(d.date()) for d in cal['missing_sessions'][:5]]} (left missing, not filled)")
    closes = decision_times(df.index, m.calendar)
    df = df[closes.to_numpy() <= pd.Timestamp.now(tz="UTC")]  # drop the bar of a session still trading
    for w in validate_prices(df, m.ticker):
        log.warning(w)
    path = store.save_prices(m.ticker, df)
    store.log_fetch("yahoo_prices", m.ticker, "ok", df, path)
    log_event(log, "prices stored", ticker=m.ticker, rows=len(df), first=str(df.index.min().date()),
              last=str(df.index.max().date()))
    return df


def ingest_source(name: str, cfg: Config, store: DataStore, client: HttpClient) -> pd.DataFrame | None:
    scfg = cfg.source(name)
    cls = CONNECTORS[name]
    missing_env = [e for e in cls.requires_env if not os.environ.get(e)]
    if missing_env:
        store.log_fetch(name, "*", "skipped", message=f"missing env {missing_env}")
        log.info(f"{name}: skipped (missing {missing_env})")
        return None
    conn = cls(client, scfg, cfg.market)
    try:
        raw = conn.fetch()
        for ds, df in raw.items():
            store.save_raw(name, ds, df)
        series = conn.transform(raw)
        if series.empty:
            store.log_fetch(name, "*", "empty", message="no data for this market")
            return None
        for w in validate_series(series, name):
            log.warning(w)
        path = store.save_series(name, series)
        store.log_fetch(name, "series", "ok", series, path, date_col="obs_date")
        log_event(log, "source stored", source=name, series=series["series_id"].nunique(), rows=len(series),
                  first=str(series["obs_date"].min().date()), last=str(series["obs_date"].max().date()))
        return series
    except Exception as exc:  # keep going: one broken provider must not stop the pipeline
        store.log_fetch(name, "*", "failed", message=f"{type(exc).__name__}: {exc}")
        log.error(f"{name}: ingestion failed: {type(exc).__name__}: {exc}")
        return None


def run_ingest(cfg: Config, sources: list[str] | None = None) -> dict[str, str]:
    store = DataStore(cfg.data_dir)
    client = make_client(cfg)
    status = {}
    ingest_prices(cfg, store, client)
    status["prices"] = "ok"
    names = sources or [n for n, s in cfg.sources.items() if s.get("enabled", False)]
    for name in names:
        if name not in CONNECTORS:
            log.warning(f"unknown source {name}")
            continue
        status[name] = "ok" if ingest_source(name, cfg, store, client) is not None else "unavailable"
    return status
