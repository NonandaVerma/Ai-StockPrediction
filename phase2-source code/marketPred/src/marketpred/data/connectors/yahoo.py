"""Yahoo Finance chart API (unofficial, keyless).

Licensing: Yahoo's terms restrict redistribution; data is used here for personal research only.
Daily bars are dated in the exchange's local time zone. Bars that are not final yet (session still
open) are dropped so that intraday runs never treat partial bars as closes.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ..base import Connector, make_series

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def parse_chart(payload: dict) -> pd.DataFrame:
    """Parse a chart API payload into an OHLCV frame indexed by local session date."""
    res = payload["chart"]["result"][0]
    tz = res["meta"].get("exchangeTimezoneName", "UTC")
    ts = res.get("timestamp") or []
    q = res["indicators"]["quote"][0]
    adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose", q["close"])
    df = pd.DataFrame({
        "open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
        "close": q.get("close"), "adj_close": adj, "volume": q.get("volume"),
    }, index=pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize())
    df.index.name = "date"
    df = df.astype(float).dropna(subset=["close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.attrs["exchange_tz"] = tz
    return df


def fetch_history(client, ticker: str, start: str = "1900-01-01") -> pd.DataFrame:
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(datetime.now(timezone.utc).timestamp())
    payload = client.get_json(CHART_URL.format(ticker=ticker),
                              params={"period1": p1, "period2": p2, "interval": "1d", "events": "div,splits"})
    if payload.get("chart", {}).get("error"):
        raise RuntimeError(f"Yahoo error for {ticker}: {payload['chart']['error']}")
    return parse_chart(payload)


def drop_unfinished(df: pd.DataFrame, close_time: str, tz: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Remove bars whose session has not closed yet (as of `now`)."""
    now = now or pd.Timestamp.now(tz="UTC")
    hh, mm = (int(x) for x in close_time.split(":"))
    closes = (df.index + pd.Timedelta(hours=hh, minutes=mm)).tz_localize(tz, nonexistent="shift_forward",
                                                                          ambiguous=False).tz_convert("UTC")
    return df[closes <= now]


class YahooCrossAsset(Connector):
    """Daily closes of cross-asset instruments, timestamped at each instrument's own close."""

    name = "yahoo_cross_asset"

    def fetch(self) -> dict[str, pd.DataFrame]:
        return {t: fetch_history(self.client, t, "1985-01-01") for t in self.cfg.get("tickers", {})}

    def transform(self, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for ticker, df in raw.items():
            rule = {"period": "D", "release_lag_days": 0, **self.cfg["tickers"][ticker]}
            df = drop_unfinished(df, rule["release_time"], rule["release_tz"])
            frames.append(make_series(f"{ticker}:close", df.index, df["close"].to_numpy(), self.name, rule))
        return pd.concat(frames, ignore_index=True)
