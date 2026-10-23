"""Local storage: immutable raw snapshots, processed Parquet datasets and a SQLite metadata log.

Layout under data_dir:
    raw/<source>/<dataset>/<UTC timestamp>.parquet   one snapshot per ingestion (never overwritten)
    processed/series/<source>.parquet                long-format point-in-time series
    processed/prices/<ticker>.parquet                OHLCV of forecast targets
    features/<market>_features.parquet               generated feature matrix
    metadata.db                                      fetch log (source, rows, hash, status, coverage)
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .base import SERIES_COLUMNS


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def frame_hash(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame (index + values), used as a dataset version id."""
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    h.update(",".join(map(str, df.columns)).encode())
    return h.hexdigest()[:12]


class DataStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        for sub in ("raw", "processed/series", "processed/prices", "features", "cache"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "metadata.db"
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS fetch_log (
                    fetched_at TEXT, source TEXT, dataset TEXT, status TEXT, rows INTEGER,
                    content_hash TEXT, min_date TEXT, max_date TEXT, path TEXT, message TEXT)"""
            )

    # ---- metadata -------------------------------------------------------------------------
    def log_fetch(self, source: str, dataset: str, status: str, df: pd.DataFrame | None = None,
                  path: Path | None = None, message: str = "", date_col: str | None = None) -> None:
        rows, h, lo, hi = 0, "", "", ""
        if df is not None and len(df):
            rows, h = len(df), frame_hash(df)
            dates = pd.to_datetime(df[date_col] if date_col else df.index, errors="coerce")
            lo, hi = str(dates.min())[:10], str(dates.max())[:10]
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO fetch_log VALUES (?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), source, dataset, status,
                 rows, h, lo, hi, str(path or ""), message[:500]),
            )

    def fetch_log(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as con:
            return pd.read_sql("SELECT * FROM fetch_log ORDER BY fetched_at", con)

    # ---- raw snapshots ---------------------------------------------------------------------
    def save_raw(self, source: str, dataset: str, df: pd.DataFrame) -> Path:
        d = self.root / "raw" / _safe(source) / _safe(dataset)
        d.mkdir(parents=True, exist_ok=True)
        path = d / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".parquet")
        df.to_parquet(path)
        return path

    def load_raw(self, source: str, dataset: str, snapshot: str = "latest") -> pd.DataFrame:
        d = self.root / "raw" / _safe(source) / _safe(dataset)
        files = sorted(d.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no raw snapshot for {source}/{dataset}")
        path = files[-1] if snapshot == "latest" else d / f"{snapshot}.parquet"
        return pd.read_parquet(path)

    # ---- processed ---------------------------------------------------------------------------
    def save_series(self, source: str, df: pd.DataFrame) -> Path:
        missing = set(SERIES_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"series frame missing columns {missing}")
        path = self.root / "processed" / "series" / f"{_safe(source)}.parquet"
        df[SERIES_COLUMNS].to_parquet(path, index=False)
        return path

    def load_series(self, sources: list[str] | None = None) -> pd.DataFrame:
        d = self.root / "processed" / "series"
        files = [d / f"{_safe(s)}.parquet" for s in sources] if sources else sorted(d.glob("*.parquet"))
        frames = [pd.read_parquet(f) for f in files if f.exists()]
        if not frames:
            return pd.DataFrame(columns=SERIES_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def save_prices(self, ticker: str, df: pd.DataFrame) -> Path:
        path = self.root / "processed" / "prices" / f"{_safe(ticker)}.parquet"
        df.to_parquet(path)
        return path

    def load_prices(self, ticker: str) -> pd.DataFrame:
        path = self.root / "processed" / "prices" / f"{_safe(ticker)}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"no prices for {ticker}; run `marketpred ingest` first")
        return pd.read_parquet(path)

    def features_path(self, market_id: str) -> Path:
        return self.root / "features" / f"{_safe(market_id)}_features.parquet"
