"""Exchange registry and trading calendar.

* Session times are exchange-local wall-clock times; conversion to UTC goes through
  `zoneinfo`, so daylight-saving changes (NYSE, LSE) are handled per date.
* Past trading sessions are taken from the market data itself (a day is a session if the
  exchange printed a bar).
* Future sessions = weekdays minus holidays listed in data/calendars/<EXCHANGE>_holidays.csv
  (columns: date,name). The project ships no holiday lists because they must come from the
  exchange's official circular; when a year is not covered, the forecast says so explicitly.
"""
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

CALENDAR_DIR = Path("data/calendars")


@dataclass(frozen=True)
class Exchange:
    code: str
    name: str
    tz: str
    open: time
    close: time
    country: str
    currency: str = ""
    currency_symbol: str = ""

    @property
    def flag(self) -> str:
        """Emoji flag built from the ISO country code (regional indicator symbols)."""
        return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in self.country.upper())

    @property
    def label(self) -> str:
        """Dropdown label, e.g. '🇮🇳 NSE · ₹ INR'."""
        return f"{self.flag} {self.code} · {self.currency_symbol} {self.currency}".strip()


EXCHANGES = {
    "NSE": Exchange("NSE", "National Stock Exchange of India", "Asia/Kolkata", time(9, 15), time(15, 30), "IN",
                    "INR", "₹"),
    "BSE": Exchange("BSE", "BSE (Bombay Stock Exchange)", "Asia/Kolkata", time(9, 15), time(15, 30), "IN",
                    "INR", "₹"),
    "NYSE": Exchange("NYSE", "New York Stock Exchange", "America/New_York", time(9, 30), time(16, 0), "US",
                     "USD", "$"),
    "NASDAQ": Exchange("NASDAQ", "Nasdaq", "America/New_York", time(9, 30), time(16, 0), "US", "USD", "$"),
    "LSE": Exchange("LSE", "London Stock Exchange", "Europe/London", time(8, 0), time(16, 30), "GB", "GBP", "£"),
}

_INDEX_EXCHANGE = {"^NSEI": "NSE", "^NSEBANK": "NSE", "^CNXIT": "NSE", "^BSESN": "BSE",
                   "^GSPC": "NYSE", "^DJI": "NYSE", "^IXIC": "NASDAQ", "^NDX": "NASDAQ", "^FTSE": "LSE"}
_SUFFIX_EXCHANGE = {".NS": "NSE", ".BO": "BSE", ".BSE": "BSE", ".L": "LSE"}


def exchange_for_symbol(symbol: str) -> tuple[Exchange, bool]:
    """Best-effort exchange from a Yahoo-style symbol. Returns (exchange, guessed)."""
    s = symbol.upper().strip()
    if s in _INDEX_EXCHANGE:
        return EXCHANGES[_INDEX_EXCHANGE[s]], False
    for suf, code in _SUFFIX_EXCHANGE.items():
        if s.endswith(suf):
            return EXCHANGES[code], False
    return EXCHANGES["NYSE"], True  # bare tickers are usually US listings; flagged as a guess


def session_time_utc(ex: Exchange, dates, anchor: str = "open") -> pd.DatetimeIndex:
    """UTC instant of the session open/close on each local date (DST-aware)."""
    t = ex.open if anchor == "open" else ex.close
    d = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().tz_localize(None)
    local = (d + pd.Timedelta(hours=t.hour, minutes=t.minute)).tz_localize(ZoneInfo(ex.tz))
    return local.tz_convert("UTC")


def is_open_now(ex: Exchange, now: pd.Timestamp | None = None, holidays: set | None = None) -> bool:
    now = (now or pd.Timestamp.now(tz="UTC")).tz_convert(ex.tz)
    if now.weekday() >= 5 or (holidays and now.normalize().tz_localize(None) in holidays):
        return False
    return ex.open <= now.time() < ex.close


def load_holidays(code: str, directory: Path = CALENDAR_DIR) -> pd.DataFrame:
    f = Path(directory) / f"{code}_holidays.csv"
    if not f.exists():
        return pd.DataFrame(columns=["date", "name"])
    h = pd.read_csv(f, parse_dates=["date"])
    h["date"] = h["date"].dt.normalize()
    return h


class TradingCalendar:
    def __init__(self, ex: Exchange, observed_sessions, holidays: pd.DataFrame | None = None):
        self.ex = ex
        self.observed = pd.DatetimeIndex(pd.to_datetime(observed_sessions)).normalize().unique().sort_values()
        self.holidays = load_holidays(ex.code) if holidays is None else holidays
        self.holiday_set = set(pd.DatetimeIndex(self.holidays["date"]))
        self.last_observed = self.observed[-1] if len(self.observed) else None

    def holiday_years_covered(self) -> set[int]:
        return set(pd.DatetimeIndex(self.holidays["date"]).year) if len(self.holidays) else set()

    def _projected(self, start, end) -> pd.DatetimeIndex:
        days = pd.bdate_range(start, end)
        return days[~days.isin(list(self.holiday_set))]

    def sessions_between(self, start, end) -> pd.DatetimeIndex:
        """Trading sessions in [start, end]: observed ones up to the last bar, projected after."""
        start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        past = self.observed[(self.observed >= start) & (self.observed <= end)]
        if self.last_observed is None or end <= self.last_observed:
            return past
        fut_start = max(start, self.last_observed + pd.Timedelta(days=1))
        return past.append(self._projected(fut_start, end))

    def next_sessions(self, after, n: int) -> pd.DatetimeIndex:
        after = pd.Timestamp(after).normalize()
        out = pd.DatetimeIndex([])
        span = max(10, n * 2)
        while len(out) < n:
            out = self.sessions_between(after + pd.Timedelta(days=1), after + pd.Timedelta(days=span))
            span *= 2
        return out[:n]

    def map_date(self, date) -> dict:
        """Map a calendar date to the trading session that actually covers it."""
        d = pd.Timestamp(date).normalize()
        ses = self.sessions_between(d, d)
        if len(ses):
            return {"requested": d, "session": d, "note": "trading session"}
        nxt = self.next_sessions(d - pd.Timedelta(days=1), 1)[0]
        if d.weekday() >= 5:
            why = "weekend"
        elif d in self.holiday_set:
            name = self.holidays.loc[self.holidays["date"] == d, "name"]
            why = f"exchange holiday ({name.iloc[0]})" if len(name) else "exchange holiday"
        elif self.last_observed is not None and d <= self.last_observed:
            why = "no trading recorded (holiday or data gap)"
        else:
            why = "non-trading day"
        return {"requested": d, "session": nxt, "note": f"{d:%a %d %b %Y} is a {why}; "
                                                         f"mapped to next session {nxt:%a %d %b %Y}"}

    def coverage_notes(self, sessions: pd.DatetimeIndex) -> list[str]:
        """Warnings for projected sessions whose holiday calendar is unknown."""
        if self.last_observed is None:
            return []
        fut = sessions[sessions > self.last_observed]
        missing = sorted(set(fut.year) - self.holiday_years_covered())
        if not missing:
            return []
        return [f"No official {self.ex.code} holiday list configured for {', '.join(map(str, missing))} "
                f"(add data/calendars/{self.ex.code}_holidays.csv). Future sessions exclude weekends only, "
                f"so an exchange holiday could shift the target session and the session count."]
