"""Keyless official / academic sources.

* US Treasury daily par yield curve (home.treasury.gov), public domain, 1990-.
* Chicago Fed National Financial Conditions Index (weekly), latest vintage only.
* University of Michigan consumer sentiment (monthly), latest file.
* BLS public API v2 (CPI, unemployment); BLS_API_KEY optional (raises daily limits).
* Shiller "ie_data" (monthly S&P earnings and dividends), academic dataset.
* Baker-Bloom-Davis daily US Economic Policy Uncertainty index (news-based).
* Caldara-Iacoviello daily Geopolitical Risk index (news-based).
* CFTC Commitments of Traders, legacy futures-only report (Socrata API).
* NASA POWER daily point data (MERRA-2 based reanalysis).
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..base import Connector, make_series


class Treasury(Connector):
    name = "treasury"
    URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
           "daily-treasury-rates.csv/{year}/all")
    TENORS = {"3 Mo": "3M", "2 Yr": "2Y", "10 Yr": "10Y", "30 Yr": "30Y"}

    def fetch(self) -> dict[str, pd.DataFrame]:
        frames = []
        for year in range(int(self.cfg.get("start_year", 1990)), datetime.now(timezone.utc).year + 1):
            params = {"type": "daily_treasury_yield_curve", "field_tdr_date_value": year, "page": "",
                      "_format": "csv"}
            content = self.client.get_bytes(self.URL.format(year=year), params=params)
            frames.append(pd.read_csv(io.BytesIO(content)))
        df = pd.concat(frames, ignore_index=True)
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        return {"par_yield_curve": df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)}

    def transform(self, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = raw["par_yield_curve"]
        rule = {"period": "D", **self.cfg}
        return pd.concat([make_series(f"treasury:{v}", df["Date"], df[k], self.name, rule)
                          for k, v in self.TENORS.items() if k in df], ignore_index=True)


class NFCI(Connector):
    name = "nfci"
    URL = "https://www.chicagofed.org/-/media/publications/nfci/nfci-data-series-csv.csv"

    def fetch(self) -> dict[str, pd.DataFrame]:
        df = pd.read_csv(io.BytesIO(self.client.get_bytes(self.URL)))
        df["Friday_of_Week"] = pd.to_datetime(df["Friday_of_Week"], format="%m/%d/%Y")
        return {"nfci": df}

    def transform(self, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = raw["nfci"]
        return pd.concat([make_series(f"nfci:{c}", df["Friday_of_Week"], df[c], self.name, self.cfg)
                          for c in ("NFCI", "ANFCI")], ignore_index=True)


class UMich(Connector):
    name = "umich"
    URL = "https://www.sca.isr.umich.edu/files/tbmics.csv"

    def fetch(self) -> dict[str, pd.DataFrame]:
        df = pd.read_csv(io.BytesIO(self.client.get_bytes(self.URL)))
        df.columns = [c.strip() for c in df.columns]
        df["date"] = pd.to_datetime(df["Month"].str.strip() + " " + df["YYYY"].astype(str), format="%B %Y")
        return {"ics": df}

    def transform(self, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = raw["ics"]
        return make_series("umich:ICS", df["date"], df["ICS_ALL"], self.name, self.cfg)


class BLS(Connector):
    name = "bls"
    URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

    def fetch(self) -> dict[str, pd.DataFrame]:
        ids = {v["id"]: k for k, v in self.cfg["series"].items()}
        start, end = int(self.cfg.get("start_year", 1988)), datetime.now(timezone.utc).year
        key = os.environ.get("BLS_API_KEY")
        step = 20 if key else 10
        rows = []
        for y0 in range(start, end + 1, step):
            body = {"seriesid": list(ids), "startyear": str(y0), "endyear": str(min(y0 + step - 1, end))}
            if key:
                body["registrationkey"] = key
            payload = self.client.get_json(self.URL, method="POST", json_body=body)
            if payload.get("status") != "REQUEST_SUCCEEDED":
                raise RuntimeError(f"BLS API: {payload.get('status')} {payload.get('message')}")
            for s in payload["Results"]["series"]:
                for obs in s["data"]:
                    if obs["period"].startswith("M") and obs["period"] != "M13":
                        rows.append((ids[s["seriesID"]], int(obs["year"]), int(obs["period"][1:]), obs["value"]))
        df = pd.DataFrame(rows, columns=["name", "year", "month", "value"])
        df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
        df["value"] = pd.to_numeric(df["value"], errors="coerce")  # BLS uses "-" for missing
        return {"bls": df.sort_values(["name", "date"]).drop_duplicates(["name", "date"]).reset_index(drop=True)}

    def transform(self, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = raw["bls"]
        frames = []
        for name, spec in self.cfg["series"].items():
            g = df[df["name"] == name]
            rule = {**self.cfg, **spec}
            frames.append(make_series(f"bls:{name}", g["date"], g["value"], self.name, rule))
        return pd.concat(frames, ignore_index=True)


class Shiller(Connector):
    name = "shiller"
    URL = "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/ie_data.xls"

    @staticmethod
    def parse(sheet: pd.DataFrame) -> pd.DataFrame:
        body = sheet.iloc[8:, :5].copy()
        body.columns = ["date", "P", "D", "E", "CPI"]
        body = body[pd.to_numeric(body["date"], errors="coerce").notna()].copy()
        x = body["date"].astype(float)
        year = np.floor(x).astype(int)
        month = np.rint((x - year) * 100).astype(int)  # Shiller encodes October as YYYY.1
        body["date"] = pd.to_datetime(dict(year=year, month=month, day=1))
        for c in ["P", "D", "E", "CPI"]:
            body[c] = pd.to_numeric(body[c], errors="coerce")
        return body.reset_index(drop=True)

    def fetch(self) -> dict[str, pd.DataFrame]:
        sheet = pd.read_excel(io.BytesIO(self.client.get_bytes(self.URL)), sheet_name="Data", header=None)
        return {"ie_data": self.parse(sheet)}

    def transform(self, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = raw["ie_data"]
        return pd.concat([make_series(f"shiller:{c}", df["date"], df[c], self.name, self.cfg)
                          for c in ("E", "D")], ignore_index=True)


class EPU(Connector):
    name = "epu"
    URL = "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv"

    def fetch(self) -> dict[str, pd.DataFrame]:
        df = pd.read_csv(io.BytesIO(self.client.get_bytes(self.URL)))
        df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day))
        return {"daily_us_epu": df}

    def transform(self, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = raw["daily_us_epu"]
        return make_series("epu:daily", df["date"], df["daily_policy_index"], self.name, self.cfg)


class GPR(Connector):
    name = "gpr"
    URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"

    def fetch(self) -> dict[str, pd.DataFrame]:
        df = pd.read_excel(io.BytesIO(self.client.get_bytes(self.URL)))
        df = df[["DAY", "GPRD", "GPRD_ACT", "GPRD_THREAT"]].dropna(subset=["DAY"])
        df["date"] = pd.to_datetime(df["DAY"].astype(int).astype(str), format="%Y%m%d")
        return {"gpr_daily": df}

    def transform(self, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = raw["gpr_daily"]
        return pd.concat([make_series(f"gpr:{c}", df["date"], df[c], self.name, self.cfg)
                          for c in ("GPRD", "GPRD_ACT", "GPRD_THREAT")], ignore_index=True)


class CFTC(Connector):
    name = "cftc"
    URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
    FIELDS = ["report_date_as_yyyy_mm_dd", "open_interest_all", "noncomm_positions_long_all",
              "noncomm_positions_short_all", "comm_positions_long_all", "comm_positions_short_all",
              "nonrept_positions_long_all", "nonrept_positions_short_all"]

    def fetch(self) -> dict[str, pd.DataFrame]:
        code = self.market.cftc_contract_code
        if not code:
            return {}
        rows, offset = [], 0
        while True:
            batch = self.client.get_json(self.URL, params={
                "$select": ",".join(self.FIELDS), "$where": f"cftc_contract_market_code='{code}'",
                "$order": "report_date_as_yyyy_mm_dd", "$limit": 5000, "$offset": offset})
            rows += batch
            if len(batch) < 5000:
                break
            offset += 5000
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.tz_localize(None).dt.normalize()
        for c in self.FIELDS[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return {f"cot_{code}": df.drop(columns=["report_date_as_yyyy_mm_dd"])}

    def transform(self, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not raw:
            return pd.DataFrame()
        df = next(iter(raw.values()))
        oi = df["open_interest_all"].replace(0, np.nan)
        series = {
            "noncomm_net_oi": (df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]) / oi,
            "comm_net_oi": (df["comm_positions_long_all"] - df["comm_positions_short_all"]) / oi,
            "nonrept_net_oi": (df["nonrept_positions_long_all"] - df["nonrept_positions_short_all"]) / oi,
            "open_interest": df["open_interest_all"],
        }
        return pd.concat([make_series(f"cftc:{k}", df["date"], v, self.name, self.cfg)
                          for k, v in series.items()], ignore_index=True)


class NasaPower(Connector):
    name = "nasa_power"
    URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

    def fetch(self) -> dict[str, pd.DataFrame]:
        loc = self.market.weather_location
        if not loc:
            return {}
        start = pd.Timestamp(self.cfg.get("start", "1984-01-01"))
        end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        frames = []
        # Request in ~10-year chunks to stay within API limits.
        for y0 in range(start.year, end.year + 1, 10):
            s, e = max(start, pd.Timestamp(f"{y0}-01-01")), min(end, pd.Timestamp(f"{y0 + 9}-12-31"))
            payload = self.client.get_json(self.URL, params={
                "parameters": ",".join(self.cfg["parameters"]), "community": "AG",
                "latitude": loc["lat"], "longitude": loc["lon"],
                "start": s.strftime("%Y%m%d"), "end": e.strftime("%Y%m%d"), "format": "JSON"})
            par = payload["properties"]["parameter"]
            fill = payload.get("header", {}).get("fill_value", -999.0)
            df = pd.DataFrame(par)
            df.index = pd.to_datetime(df.index, format="%Y%m%d")
            frames.append(df.replace(fill, np.nan))
        df = pd.concat(frames).sort_index()
        df.index.name = "date"
        return {f"point_{loc['name']}": df[~df.index.duplicated()]}

    def transform(self, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not raw:
            return pd.DataFrame()
        df = next(iter(raw.values()))
        return pd.concat([make_series(f"weather:{c}", df.index, df[c].to_numpy(), self.name, self.cfg)
                          for c in df.columns], ignore_index=True)


class FRED(Connector):
    """Optional: FRED/ALFRED with real-time vintages. Requires FRED_API_KEY (not used by default).

    Each observation is emitted once per vintage with available_at = realtime_start at 23:59 ET
    (conservatively unusable at that day's close). pit.align/pit.derive handle multiple vintages.
    """

    name = "fred"
    requires_env = ("FRED_API_KEY",)
    URL = "https://api.stlouisfed.org/fred/series/observations"

    def fetch(self) -> dict[str, pd.DataFrame]:
        key = os.environ.get("FRED_API_KEY")
        if not key:
            raise RuntimeError("FRED_API_KEY not set")
        out = {}
        for sid in self.cfg.get("series", []):
            payload = self.client.get_json(self.URL, params={
                "series_id": sid, "api_key": key, "file_type": "json",
                "realtime_start": "1776-07-04", "realtime_end": "9999-12-31"})
            out[sid] = pd.DataFrame(payload["observations"])
        return out

    def transform(self, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for sid, df in raw.items():
            s = make_series(f"fred:{sid}", pd.to_datetime(df["date"]), df["value"].replace(".", np.nan),
                            self.name, {"period": "D", "release_time": "00:00", "release_tz": "UTC"},
                            revised=True)
            vint = pd.to_datetime(df.loc[pd.to_numeric(df["value"], errors="coerce").notna(), "realtime_start"])
            local = (pd.DatetimeIndex(vint) + pd.Timedelta(hours=23, minutes=59)).tz_localize("America/New_York")
            s["available_at"] = local.tz_convert("UTC")
            frames.append(s)
        return pd.concat(frames, ignore_index=True)
