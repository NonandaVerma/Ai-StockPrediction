"""Data-access functions for the SQLite schema in migrations/. All SQL lives here."""
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd

BAR_COLS = ["open", "high", "low", "close", "adj_close", "volume"]


def utc_iso(ts) -> str:
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def now_iso() -> str:
    return utc_iso(pd.Timestamp.now(tz="UTC"))


def stable_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _num(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return None
    return float(v)


# ---------------------------------------------------------------- sources / instruments
def source_id(con, name: str, kind: str = "market", realtime: bool = False, notes: str = "") -> int:
    con.execute("INSERT INTO data_sources(name, kind, realtime, notes) VALUES (?,?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET realtime=excluded.realtime, notes=excluded.notes",
                (name, kind, int(realtime), notes))
    con.commit()
    return con.execute("SELECT id FROM data_sources WHERE name=?", (name,)).fetchone()[0]


def upsert_instrument(con, symbol, exchange, name=None, sector=None, currency=None, kind=None) -> int:
    con.execute("""INSERT INTO instruments(symbol, exchange, name, sector, currency, kind) VALUES (?,?,?,?,?,?)
                   ON CONFLICT(symbol) DO UPDATE SET exchange=excluded.exchange,
                     name=COALESCE(excluded.name, instruments.name),
                     sector=COALESCE(excluded.sector, instruments.sector),
                     currency=COALESCE(excluded.currency, instruments.currency),
                     kind=COALESCE(excluded.kind, instruments.kind),
                     updated_at=strftime('%Y-%m-%dT%H:%M:%S+00:00','now')""",
                (symbol, exchange, name, sector, currency, kind))
    con.commit()
    return con.execute("SELECT id FROM instruments WHERE symbol=?", (symbol,)).fetchone()[0]


def get_instrument(con, symbol) -> dict | None:
    r = con.execute("SELECT * FROM instruments WHERE symbol=?", (symbol,)).fetchone()
    return dict(r) if r else None


def list_instruments(con) -> pd.DataFrame:
    return pd.read_sql_query("SELECT symbol, name, exchange, sector, currency, kind FROM instruments "
                             "ORDER BY symbol", con)


# ---------------------------------------------------------------- price bars
def save_bars(con, instrument_id: int, source: int, interval: str, df: pd.DataFrame) -> int:
    """Upsert bars. Re-imports update values in place (providers revise adjusted closes)."""
    if df.empty:
        return 0
    fetched = now_iso()
    rows = [(instrument_id, source, interval, d.strftime("%Y-%m-%d"), utc_iso(r.ts_utc),
             *(_num(getattr(r, c)) for c in BAR_COLS), fetched)
            for d, r in zip(df.index, df.itertuples(index=False))]
    con.executemany(f"""INSERT INTO price_bars(instrument_id, source_id, interval, session_date, ts_utc,
                        {', '.join(BAR_COLS)}, fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(instrument_id, interval, ts_utc) DO UPDATE SET
                        source_id=excluded.source_id, open=excluded.open, high=excluded.high,
                        low=excluded.low, close=excluded.close, adj_close=excluded.adj_close,
                        volume=excluded.volume, fetched_at=excluded.fetched_at""", rows)
    con.commit()
    return len(rows)


def load_bars(con, instrument_id: int, interval: str, start=None, end=None) -> pd.DataFrame:
    q = ("SELECT session_date, ts_utc, open, high, low, close, adj_close, volume FROM price_bars "
         "WHERE instrument_id=? AND interval=?")
    args = [instrument_id, interval]
    if start is not None:
        q += " AND session_date >= ?"; args.append(pd.Timestamp(start).strftime("%Y-%m-%d"))
    if end is not None:
        q += " AND session_date <= ?"; args.append(pd.Timestamp(end).strftime("%Y-%m-%d"))
    df = pd.read_sql_query(q + " ORDER BY ts_utc", con, params=args)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    if interval == "1d":
        df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("session_date")), name="date")
    else:
        df.pop("session_date")
        df.index = pd.DatetimeIndex(df["ts_utc"], name="date")
    return df


def set_freshness(con, instrument_id, interval, source, first_bar, last_bar, label):
    con.execute("""INSERT INTO data_freshness VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(instrument_id, interval, source_id) DO UPDATE SET
                   first_bar_utc=excluded.first_bar_utc, last_bar_utc=excluded.last_bar_utc,
                   fetched_at=excluded.fetched_at, label=excluded.label""",
                (instrument_id, interval, source, utc_iso(first_bar), utc_iso(last_bar), now_iso(), label))
    con.commit()


def get_freshness(con, instrument_id, interval) -> dict | None:
    r = con.execute("""SELECT f.*, s.name AS source, s.notes AS source_notes FROM data_freshness f
                       JOIN data_sources s ON s.id=f.source_id WHERE instrument_id=? AND interval=?
                       ORDER BY fetched_at DESC LIMIT 1""", (instrument_id, interval)).fetchone()
    return dict(r) if r else None


# ---------------------------------------------------------------- astronomy / astrology
def save_positions(con, long: pd.DataFrame, zodiac: str, ayanamsa: str, ephemeris: str) -> None:
    rows = [(utc_iso(r.ts_utc), zodiac, ayanamsa, r.body, r.longitude, r.latitude, r.distance_au,
             r.speed_deg_day, r.declination, ephemeris) for r in long.itertuples(index=False)]
    con.executemany("""INSERT OR IGNORE INTO planetary_positions(ts_utc, zodiac, ayanamsa, body, longitude,
                       latitude, distance_au, speed_deg_day, declination, ephemeris)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""", rows)
    con.commit()


def save_events(con, events: pd.DataFrame, zodiac: str, ayanamsa: str) -> None:
    rows = [(pd.Timestamp(r.date).strftime("%Y-%m-%d"), r.event_type, r.body1, r.body2 or "", r.detail,
             zodiac, ayanamsa) for r in events.itertuples(index=False)]
    con.executemany("""INSERT OR IGNORE INTO astro_events(event_date, event_type, body1, body2, detail,
                       zodiac, ayanamsa) VALUES (?,?,?,?,?,?,?)""", rows)
    con.commit()


# ---------------------------------------------------------------- models / backtests / predictions
def register_model(con, name: str, version: str, family: str, params: dict) -> int:
    con.execute("INSERT OR IGNORE INTO model_versions(name, version, family, params_json) VALUES (?,?,?,?)",
                (name, version, family, json.dumps(params, sort_keys=True, default=str)))
    con.commit()
    return con.execute("SELECT id FROM model_versions WHERE name=? AND version=?", (name, version)).fetchone()[0]


def save_backtest(con, run_key, instrument_id, model_version_id, gap, horizon, zodiac, ayanamsa,
                  data_start, data_end, eval_start, eval_end, folds, cost_bps, metrics: dict) -> bool:
    """Returns False when an identical run (same run_key) is already stored."""
    cur = con.execute("""INSERT OR IGNORE INTO backtest_runs(run_key, instrument_id, model_version_id,
                         gap_sessions, horizon_sessions, zodiac, ayanamsa, data_start, data_end, eval_start,
                         eval_end, folds, cost_bps, metrics_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                      (run_key, instrument_id, model_version_id, gap, horizon, zodiac, ayanamsa,
                       str(data_start)[:10], str(data_end)[:10], str(eval_start)[:10], str(eval_end)[:10],
                       folds, cost_bps, json.dumps(metrics, default=float)))
    con.commit()
    return cur.rowcount == 1


def list_backtests(con, symbol: str | None = None) -> pd.DataFrame:
    q = """SELECT b.created_at, i.symbol, m.name AS model, m.version, m.family, b.gap_sessions,
                  b.horizon_sessions, b.zodiac, b.ayanamsa, b.eval_start, b.eval_end, b.folds, b.cost_bps,
                  b.metrics_json FROM backtest_runs b JOIN instruments i ON i.id=b.instrument_id
           JOIN model_versions m ON m.id=b.model_version_id"""
    args = []
    if symbol:
        q += " WHERE i.symbol=?"; args.append(symbol)
    df = pd.read_sql_query(q + " ORDER BY b.created_at DESC", con, params=args)
    if len(df):
        metrics = pd.json_normalize(df.pop("metrics_json").map(json.loads))
        df = pd.concat([df, metrics.drop(columns=[c for c in metrics if c in df])], axis=1)
    return df


def save_prediction(con, row: dict) -> bool:
    """Insert a forecast; returns False if the same forecast already exists (no duplicates)."""
    cols = list(row)
    cur = con.execute(f"INSERT OR IGNORE INTO predictions({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                      [json.dumps(v, default=str) if isinstance(v, (dict, list)) else v for v in row.values()])
    con.commit()
    return cur.rowcount == 1


def list_predictions(con, symbol: str | None = None) -> pd.DataFrame:
    q = """SELECT p.id, p.generated_at, i.symbol, m.name AS model, p.as_of_session, p.window_start,
                  p.window_end, p.horizon_sessions, p.zodiac, p.direction, p.prob_up, p.expected_return,
                  p.interval_low, p.interval_high, p.risk_category, p.confidence, p.evidence, p.realized_return
           FROM predictions p JOIN instruments i ON i.id=p.instrument_id
           JOIN model_versions m ON m.id=p.model_version_id"""
    args = []
    if symbol:
        q += " WHERE i.symbol=?"; args.append(symbol)
    return pd.read_sql_query(q + " ORDER BY p.generated_at DESC", con, params=args)


def set_realized(con, prediction_id: int, realized: float) -> None:
    con.execute("UPDATE predictions SET realized_return=? WHERE id=?", (realized, prediction_id))
    con.commit()


# ---------------------------------------------------------------- help texts (UI tooltips / glossary)
def help_texts(con) -> dict[str, str]:
    """key -> description, for tooltips."""
    return {r[0]: r[1] for r in con.execute("SELECT key, description FROM help_texts")}


def list_help(con) -> pd.DataFrame:
    return pd.read_sql_query("SELECT key, term, category, description, updated_at FROM help_texts "
                             "ORDER BY category, term", con)


def save_help(con, rows: pd.DataFrame) -> int:
    """Insert or update help texts (columns: key, term, category, description). Returns rows written."""
    rows = rows.dropna(subset=["key", "description"])
    con.executemany("""INSERT INTO help_texts(key, term, category, description) VALUES (?,?,?,?)
                       ON CONFLICT(key) DO UPDATE SET term=excluded.term, category=excluded.category,
                         description=excluded.description,
                         updated_at=strftime('%Y-%m-%dT%H:%M:%S+00:00','now')""",
                    [(str(r.key).strip().lower(), r.term or r.key, r.category or "app", r.description)
                     for r in rows.itertuples()])
    con.commit()
    return len(rows)


def delete_help(con, keys) -> None:
    con.executemany("DELETE FROM help_texts WHERE key=?", [(k,) for k in keys])
    con.commit()


def table_counts(con) -> dict:
    names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' "
                                       "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    return {n: con.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in names}


def connect_from_cfg(cfg) -> sqlite3.Connection:
    from .db import connect
    return connect((cfg.get("database") or {}).get("path", "data/astromarket.db"))
