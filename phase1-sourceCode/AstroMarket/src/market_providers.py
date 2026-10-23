"""Provider-independent market data layer.

Every provider returns the same frame: index = exchange-local session date (naive),
columns = ts_utc, open, high, low, close, adj_close, volume. The rest of the app only talks to
`MarketDataProvider`, so a provider can be swapped without touching the prediction engine.

Rules: never fabricate prices (missing data raises ProviderError), never call data real-time
unless the provider guarantees it, never log or display credentials.
"""
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps

import numpy as np
import pandas as pd

from .exchanges import Exchange, exchange_for_symbol, session_time_utc

log = logging.getLogger(__name__)
COLUMNS = ["ts_utc", "open", "high", "low", "close", "adj_close", "volume"]
INTRADAY_LIMIT_DAYS = {"5m": 59, "15m": 59, "1h": 729}   # Yahoo look-back limits


class ProviderError(RuntimeError):
    """Data could not be obtained. Callers must surface this, not substitute values."""


@dataclass
class InstrumentInfo:
    symbol: str
    name: str | None
    exchange: str
    exchange_guessed: bool
    sector: str | None
    currency: str | None
    kind: str


def retry(fn):
    """Exponential backoff for transient failures. ProviderError (bad symbol, no data) is not retried."""
    @wraps(fn)
    def wrapper(self, *a, **kw):
        err = None
        for attempt in range(self.retries):
            self._throttle()
            try:
                return fn(self, *a, **kw)
            except ProviderError:
                raise
            except Exception as e:  # network errors, timeouts, HTTP 429/5xx
                err = e
                wait = 2 ** attempt
                log.warning("%s.%s failed (%s: %s); retry %d/%d in %ss", self.name, fn.__name__,
                            type(e).__name__, str(e)[:120], attempt + 1, self.retries, wait)
                time.sleep(wait)
        raise ProviderError(f"{self.name}: {fn.__name__} failed after {self.retries} attempts "
                            f"({type(err).__name__}: {str(err)[:200]})")
    return wrapper


class MarketDataProvider(ABC):
    name = "base"
    realtime = False            # True only for feeds contractually delivering real-time quotes
    notes = ""

    def __init__(self, timeout_s=20, retries=3, min_interval_s=0.5):
        self.timeout_s, self.retries, self.min_interval_s = timeout_s, max(1, retries), min_interval_s
        self._last_call = 0.0

    def _throttle(self):
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    @abstractmethod
    def history(self, symbol: str, start, end, interval: str = "1d") -> pd.DataFrame: ...

    def instrument_info(self, symbol: str) -> InstrumentInfo:
        ex, guessed = exchange_for_symbol(symbol)
        return InstrumentInfo(symbol, None, ex.code, guessed, None, None,
                              "index" if symbol.startswith("^") else "equity")

    def search(self, query: str) -> list[dict]:
        return []


def _finish(df: pd.DataFrame, ex: Exchange, interval: str) -> pd.DataFrame:
    """Normalise a provider frame: local session-date index + UTC timestamp column."""
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    idx = pd.DatetimeIndex(df.index)
    if interval == "1d":
        local_dates = (idx.tz_convert(ex.tz) if idx.tz is not None else idx).tz_localize(None).normalize()
        df = df.set_axis(local_dates)
        df["ts_utc"] = session_time_utc(ex, local_dates, "close")   # a daily close is known at session close
    else:
        utc = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize(ex.tz).tz_convert("UTC")
        df["ts_utc"] = utc
        df = df.set_axis(utc.tz_convert(ex.tz).tz_localize(None))
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["close"])[COLUMNS]


class YahooProvider(MarketDataProvider):
    name = "yahoo"
    notes = ("Unofficial Yahoo Finance feed via yfinance. No delivery guarantee; quotes can lag "
             "(typically up to ~15 min, exchange-dependent). Never treated as real-time.")

    @retry
    def history(self, symbol, start, end, interval="1d"):
        import yfinance as yf
        if interval in INTRADAY_LIMIT_DAYS:
            earliest = pd.Timestamp.today().normalize() - pd.Timedelta(days=INTRADAY_LIMIT_DAYS[interval])
            start = max(pd.Timestamp(start), earliest)
        end = pd.Timestamp(end) + pd.Timedelta(days=1)            # yfinance end is exclusive
        raw = yf.Ticker(symbol).history(start=pd.Timestamp(start).strftime("%Y-%m-%d"),
                                        end=end.strftime("%Y-%m-%d"), interval=interval,
                                        auto_adjust=False, actions=False, timeout=self.timeout_s)
        if raw is None or raw.empty:
            raise ProviderError(f"yahoo returned no {interval} data for {symbol} "
                                f"between {pd.Timestamp(start).date()} and {end.date()}")
        df = raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                                 "Adj Close": "adj_close", "Volume": "volume"})
        if "adj_close" not in df:
            df["adj_close"] = pd.NA
        ex, _ = exchange_for_symbol(symbol)
        return _finish(df, ex, interval)

    @retry
    def instrument_info(self, symbol):
        import yfinance as yf
        base = super().instrument_info(symbol)
        try:
            info = yf.Ticker(symbol).get_info() or {}
        except Exception as e:  # info endpoint is flaky; metadata is optional
            log.warning("yahoo info for %s unavailable: %s", symbol, e)
            return base
        base.name = info.get("longName") or info.get("shortName")
        base.sector = info.get("sector")
        base.currency = info.get("currency")
        qt = (info.get("quoteType") or "").lower()
        base.kind = {"index": "index", "equity": "equity", "etf": "etf"}.get(qt, base.kind)
        return base

    def search(self, query):
        import yfinance as yf
        try:
            res = yf.Search(query, max_results=10, timeout=self.timeout_s).quotes
        except Exception as e:
            log.warning("yahoo search failed: %s", e)
            return []
        return [{"symbol": q.get("symbol"), "name": q.get("longname") or q.get("shortname"),
                 "exchange": q.get("exchDisp") or q.get("exchange"), "type": q.get("quoteType")}
                for q in res if q.get("symbol")]


class AlphaVantageProvider(MarketDataProvider):
    """Alpha Vantage daily bars. Requires env var ALPHAVANTAGE_API_KEY.
    Symbols use Alpha Vantage notation (e.g. IBM, RELIANCE.BSE)."""
    name = "alphavantage"
    notes = "Alpha Vantage REST API, end-of-day daily bars (free tier: 25 requests/day)."
    URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str | None = None, **kw):
        super().__init__(**kw)
        self._key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")

    @retry
    def history(self, symbol, start, end, interval="1d"):
        if interval != "1d":
            raise ProviderError("alphavantage provider supports daily bars only")
        if not self._key:
            raise ProviderError("ALPHAVANTAGE_API_KEY environment variable is not set")
        q = urllib.parse.urlencode({"function": "TIME_SERIES_DAILY", "symbol": symbol,
                                    "outputsize": "full", "apikey": self._key})
        with urllib.request.urlopen(f"{self.URL}?{q}", timeout=self.timeout_s) as r:   # key never logged
            payload = json.loads(r.read().decode())
        series = payload.get("Time Series (Daily)")
        if not series:
            msg = payload.get("Error Message") or payload.get("Note") or payload.get("Information") or "no data"
            raise ProviderError(f"alphavantage: {str(msg)[:200]}")
        df = pd.DataFrame.from_dict(series, orient="index").astype(float)
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()                     # API returns newest first
        df["adj_close"] = pd.NA      # adjusted series is a premium endpoint; not fabricated
        df = df.loc[str(pd.Timestamp(start).date()):str(pd.Timestamp(end).date())]
        if df.empty:
            raise ProviderError(f"alphavantage returned no rows for {symbol} in the requested range")
        ex, _ = exchange_for_symbol(symbol)
        return _finish(df, ex, interval)


def _synthetic_prices(start, end, seed, signal=0.0):
    """Random-walk index with volatility clustering and NO astro effect,
    unless `signal` > 0, which adds a fake Moon-phase effect (positive control)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    vol = np.empty(n)
    vol[0] = 0.012
    shocks = rng.standard_normal(n)
    for t in range(1, n):  # simple GARCH(1,1)-like volatility
        vol[t] = np.sqrt(0.000002 + 0.08 * (vol[t-1] * shocks[t-1]) ** 2 + 0.90 * vol[t-1] ** 2)
    rets = 0.0004 + vol * shocks
    if signal > 0:
        # Fake effect: returns higher in waxing Moon. Synodic month = 29.530589 days.
        days = (dates - pd.Timestamp("2000-01-06")).days.values  # 2000-01-06 was a new moon
        phase = 2 * np.pi * (days % 29.530589) / 29.530589
        rets = rets + signal * np.sin(phase)
    close = 1000 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                       "volume": 1e6}, index=dates)
    df.index.name = "date"
    return df


class SyntheticProvider(MarketDataProvider):
    """Fake random-walk prices for offline testing. Always labelled synthetic."""
    name = "synthetic"
    notes = "Simulated prices (GARCH-like random walk). Not market data."

    def __init__(self, seed=42, signal=0.0, **kw):
        super().__init__(**kw)
        self.seed, self.signal = seed, signal

    def history(self, symbol, start, end, interval="1d"):
        if interval != "1d":
            raise ProviderError("synthetic provider supports daily bars only")
        df = _synthetic_prices(pd.Timestamp(start), pd.Timestamp(end), self.seed, self.signal)
        df["adj_close"] = df["close"]
        ex, _ = exchange_for_symbol(symbol)
        return _finish(df, ex, interval)


def get_provider(cfg, name: str | None = None) -> MarketDataProvider:
    p = cfg.get("providers", {}) or {}
    kw = dict(timeout_s=p.get("timeout_s", 20), retries=p.get("retries", 3),
              min_interval_s=p.get("min_interval_s", 0.5))
    name = name or ("synthetic" if cfg["market"].get("synthetic") else p.get("market", "yahoo"))
    if name == "yahoo":
        return YahooProvider(**kw)
    if name == "alphavantage":
        return AlphaVantageProvider(**kw)
    if name == "synthetic":
        return SyntheticProvider(seed=cfg.get("seed", 42), signal=cfg["market"].get("synthetic_signal", 0.0), **kw)
    raise ValueError(f"unknown market data provider '{name}'")
