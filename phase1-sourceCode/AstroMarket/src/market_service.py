"""Market data service: provider -> SQLite cache -> callers, with freshness labelling.

Freshness labels (never upgraded beyond what the source guarantees):
  real-time   provider declares real-time AND the last bar is < 2 minutes old during market hours
  delayed     intraday data from a provider without a real-time guarantee, or a stale real-time feed
  end-of-day  daily bars, or any data while the exchange is closed
  synthetic   simulated prices
"""
import logging

import numpy as np
import pandas as pd

from . import repository as repo
from .exchanges import EXCHANGES, TradingCalendar, exchange_for_symbol, is_open_now, load_holidays
from .market_providers import ProviderError, get_provider

log = logging.getLogger(__name__)
REFRESH_AFTER = pd.Timedelta(hours=1)


def freshness_label(provider_name: str, realtime: bool, interval: str, last_bar_utc, ex, now=None) -> str:
    now = now or pd.Timestamp.now(tz="UTC")
    if provider_name == "synthetic":
        return "synthetic"
    if interval == "1d" or not is_open_now(ex, now):
        return "end-of-day"
    if realtime and (now - pd.Timestamp(last_bar_utc)) < pd.Timedelta(minutes=2):
        return "real-time"
    return "delayed"


class MarketService:
    def __init__(self, cfg, con=None, provider=None):
        self.cfg = cfg
        self.con = con or repo.connect_from_cfg(cfg)
        self.provider = provider or get_provider(cfg)
        self.source = repo.source_id(self.con, self.provider.name, "market", self.provider.realtime,
                                     self.provider.notes)
        self.history_start = (cfg.get("forecast") or {}).get("history_start", cfg["market"]["start"])

    # ------------------------------------------------------------ instruments
    def instrument(self, symbol: str, exchange: str | None = None) -> dict:
        row = repo.get_instrument(self.con, symbol)
        if row and (exchange is None or row["exchange"] == exchange):
            return row
        info = self.provider.instrument_info(symbol)
        repo.upsert_instrument(self.con, symbol, exchange or info.exchange, info.name, info.sector,
                               info.currency, info.kind)
        row = repo.get_instrument(self.con, symbol)
        row["exchange_guessed"] = info.exchange_guessed and exchange is None
        return row

    def exchange(self, symbol: str):
        row = self.instrument(symbol)
        return EXCHANGES.get(row["exchange"]) or exchange_for_symbol(symbol)[0]

    def calendar(self, symbol: str, sessions=None) -> TradingCalendar:
        ex = self.exchange(symbol)
        sessions = self.daily(symbol).index if sessions is None else sessions
        return TradingCalendar(ex, sessions, load_holidays(ex.code))

    # ------------------------------------------------------------ bars
    def _drop_incomplete_session(self, df, ex):
        """Yahoo returns today's in-progress session as a daily bar; it is not a close yet."""
        if len(df) and is_open_now(ex):
            today = pd.Timestamp.now(tz=ex.tz).tz_localize(None).normalize()
            df = df[df.index < today]
        return df

    def daily(self, symbol: str, refresh: bool = False) -> pd.DataFrame:
        """Full daily history (cached). Adds pct_change based on adjusted close where available."""
        inst = self.instrument(symbol)
        ex = EXCHANGES[inst["exchange"]]
        cached = repo.load_bars(self.con, inst["id"], "1d")
        fresh = repo.get_freshness(self.con, inst["id"], "1d")
        stale = fresh is None or (pd.Timestamp.now(tz="UTC") - pd.Timestamp(fresh["fetched_at"])) > REFRESH_AFTER
        if cached.empty or refresh or stale:
            start = self.history_start if cached.empty else cached.index[-1] - pd.Timedelta(days=10)
            try:
                new = self.provider.history(symbol, start, pd.Timestamp.today(), "1d")
                new = self._drop_incomplete_session(new, ex)
                repo.save_bars(self.con, inst["id"], self.source, "1d", new)
                cached = repo.load_bars(self.con, inst["id"], "1d")
                if len(cached):
                    repo.set_freshness(self.con, inst["id"], "1d", self.source, cached.ts_utc.iloc[0],
                                       cached.ts_utc.iloc[-1],
                                       freshness_label(self.provider.name, False, "1d", None, ex))
            except ProviderError as e:
                if cached.empty:
                    raise
                log.warning("refresh failed for %s, serving cached data: %s", symbol, e)
        if cached.empty:
            raise ProviderError(f"No daily data available for {symbol}")
        return add_returns(cached)

    def intraday(self, symbol: str, interval: str = "15m", days: int = 5) -> pd.DataFrame:
        inst = self.instrument(symbol)
        start = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
        df = self.provider.history(symbol, start, pd.Timestamp.today(), interval)
        repo.save_bars(self.con, inst["id"], self.source, interval, df)
        ex = EXCHANGES[inst["exchange"]]
        repo.set_freshness(self.con, inst["id"], interval, self.source, df.ts_utc.iloc[0], df.ts_utc.iloc[-1],
                           freshness_label(self.provider.name, self.provider.realtime, interval,
                                           df.ts_utc.iloc[-1], ex))
        return df

    def latest(self, symbol: str) -> dict:
        """Latest available price with an honest freshness label."""
        inst = self.instrument(symbol)
        ex = EXCHANGES[inst["exchange"]]
        now = pd.Timestamp.now(tz="UTC")
        bar, interval = None, "1d"
        if is_open_now(ex) and self.provider.name != "synthetic":
            try:
                intr = self.intraday(symbol, "5m", 2)
                bar, interval = intr.iloc[-1], "5m"
            except ProviderError as e:
                log.warning("intraday quote unavailable for %s: %s", symbol, e)
        daily = self.daily(symbol)
        if bar is None:
            bar = daily.iloc[-1]
        prev_close = daily["close"].iloc[-1] if interval == "5m" else daily["close"].iloc[-2]
        last_ts = pd.Timestamp(bar["ts_utc"])
        return {"symbol": symbol, "price": float(bar["close"]), "prev_close": float(prev_close),
                "change_pct": float(bar["close"] / prev_close - 1), "ts_utc": last_ts,
                "ts_exchange": last_ts.tz_convert(ex.tz), "age": now - last_ts,
                "label": freshness_label(self.provider.name, self.provider.realtime, interval, last_ts, ex, now),
                "source": self.provider.name, "source_notes": self.provider.notes,
                "market_open": is_open_now(ex, now), "currency": inst.get("currency")}


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    px = df["adj_close"].where(df["adj_close"].notna(), df["close"]).astype(float)
    df["price"] = px                       # adjusted close when available, else close
    df["pct_change"] = px.pct_change()
    df["log_ret"] = np.log(px).diff()
    return df


def resample_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Daily -> weekly ('W-FRI') / monthly ('ME') / yearly ('YE') OHLCV bars from actual sessions only."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "price": "last", "volume": "sum"}
    out = df[list(agg)].resample(rule).agg(agg).dropna(subset=["close"])
    out["pct_change"] = out["price"].pct_change()
    return out
