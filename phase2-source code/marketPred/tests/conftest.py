"""Shared fixtures: synthetic, fully offline data (tests never touch the network unless marked)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketpred.calendar import trading_days
from marketpred.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def sessions() -> pd.DatetimeIndex:
    return trading_days("NYSE", "2015-01-01", "2020-12-31")


@pytest.fixture()
def prices(sessions) -> pd.DataFrame:
    """Random-walk OHLCV on real NYSE sessions."""
    rng = np.random.default_rng(0)
    n = len(sessions)
    r = rng.normal(0.0003, 0.01, n)
    close = 100 * np.exp(np.cumsum(r))
    high = close * np.exp(np.abs(rng.normal(0, 0.005, n)))
    low = close * np.exp(-np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "adj_close": close,
                         "volume": rng.integers(1_000_000, 2_000_000, n).astype(float)}, index=sessions)
