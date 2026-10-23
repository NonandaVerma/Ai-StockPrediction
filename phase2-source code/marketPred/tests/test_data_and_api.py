"""Connector parsing (offline payloads), storage round-trips, config validation, API."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketpred.config import load_config
from marketpred.data.connectors.official import Shiller
from marketpred.data.connectors.yahoo import drop_unfinished, parse_chart
from marketpred.data.store import DataStore
from marketpred.data.validation import DataValidationError, validate_prices


def test_parse_yahoo_chart_uses_exchange_local_dates():
    payload = {"chart": {"result": [{
        "meta": {"exchangeTimezoneName": "America/New_York"},
        "timestamp": [1704205800, 1704292200],  # 2024-01-02/03 14:30 UTC (09:30 ET)
        "indicators": {"quote": [{"open": [1, 2], "high": [2, 3], "low": [0.5, 1.5], "close": [1.5, None],
                                  "volume": [10, 20]}], "adjclose": [{"adjclose": [1.4, None]}]},
    }], "error": None}}
    df = parse_chart(payload)
    assert list(df.index) == [pd.Timestamp("2024-01-02")]  # null close dropped, not filled
    assert df["adj_close"].iloc[0] == 1.4


def test_drop_unfinished_bar():
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]))
    now = pd.Timestamp("2024-01-03 18:00", tz="UTC")  # 13:00 ET, session still open
    assert len(drop_unfinished(df, "16:00", "America/New_York", now)) == 1


def test_shiller_date_encoding():
    sheet = pd.DataFrame([[None] * 5] * 8 + [[2024.01, 4800, 70, 190, 308], [2024.1, 5700, 72, 200, 315],
                                             ["note", None, None, None, None]])
    out = Shiller.parse(sheet)
    assert list(out["date"]) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-10-01")]


def test_store_roundtrip_and_fetch_log(tmp_path):
    st = DataStore(tmp_path)
    df = pd.DataFrame({"a": [1, 2]}, index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"]))
    st.save_raw("src", "ds", df)
    pd.testing.assert_frame_equal(st.load_raw("src", "ds"), df, check_freq=False)
    st.log_fetch("src", "ds", "ok", df)
    assert st.fetch_log()["rows"].iloc[0] == 2


def test_validate_prices_rejects_bad_index(prices):
    with pytest.raises(DataValidationError):
        validate_prices(prices.iloc[::-1])
    assert validate_prices(prices) == []


def test_config_override_and_validation():
    cfg = load_config(overrides={"targets": {"horizons": [1]}})
    assert cfg.targets.horizons == [1] and cfg.market.ticker == "^GSPC"
    with pytest.raises(Exception):
        load_config(overrides={"validation": {"holdout_start": "1999-01-01"}})


def test_api_health_and_missing_artifacts(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MARKETPRED_ARTIFACTS_DIR", str(tmp_path))
    from marketpred.api.app import app

    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    assert c.get("/forecast/latest").status_code == 404
    assert c.get("/results/not_a_table").status_code in (404,)
