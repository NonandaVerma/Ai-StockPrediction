"""Timestamp alignment and point-in-time correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketpred.calendar import decision_times
from marketpred.data import pit
from marketpred.data.base import compute_available_at, make_series
from marketpred.data.validation import DataValidationError, validate_series


def test_decision_times_handle_dst_and_early_close():
    d = pd.DatetimeIndex(["2020-01-15", "2020-07-15", "2020-11-27"])  # winter, summer, day after Thanksgiving
    ts = decision_times(d, "NYSE")
    assert str(ts.iloc[0]) == "2020-01-15 21:00:00+00:00"  # 16:00 EST
    assert str(ts.iloc[1]) == "2020-07-15 20:00:00+00:00"  # 16:00 EDT
    assert str(ts.iloc[2]) == "2020-11-27 18:00:00+00:00"  # 13:00 EST early close


def test_decision_times_reject_non_sessions():
    with pytest.raises(ValueError):
        decision_times(pd.DatetimeIndex(["2020-12-25"]), "NYSE")


def test_available_at_monthly_release_lag():
    a = compute_available_at(pd.DatetimeIndex(["2020-01-01"]), "M", 35, "08:30", "America/New_York")
    # Jan 2020 period ends Jan 31; +35 days = Mar 6 08:30 EST = 13:30 UTC
    assert str(a[0]) == "2020-03-06 13:30:00+00:00"


def test_available_at_daily_is_dst_aware():
    a = compute_available_at(pd.DatetimeIndex(["2020-03-06", "2020-03-09"]), "D", 0, "16:15", "America/New_York")
    assert str(a[0]) == "2020-03-06 21:15:00+00:00" and str(a[1]) == "2020-03-09 20:15:00+00:00"


def test_make_series_ignores_input_index_labels():
    """Regression test: values passed as a Series with a non-default index must not be re-aligned."""
    vals = pd.Series([1.0, 2.0, 3.0], index=[10, 11, 12])
    s = make_series("x", pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"]), vals, "src", {"period": "D"})
    assert s["value"].tolist() == [1.0, 2.0, 3.0]


def _series(obs, vals, avail):
    return pd.DataFrame({"series_id": "x", "obs_date": pd.DatetimeIndex(obs), "value": vals,
                         "available_at": pd.DatetimeIndex(avail, tz="UTC"), "source": "t", "revised": False})


def test_align_excludes_values_published_after_decision():
    df = _series(["2020-01-02", "2020-01-03"], [1.0, 2.0], ["2020-01-02 21:15", "2020-01-03 21:15"])
    dec = pd.Series(pd.DatetimeIndex(["2020-01-02 21:00", "2020-01-03 21:15", "2020-01-06 21:00"], tz="UTC"),
                    index=pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"]))
    out = pit.align(df, dec)
    assert np.isnan(out["value"].iloc[0])  # published 15 min after the close -> unusable that day
    assert out["value"].iloc[1] == 2.0      # published exactly at decision time -> usable
    assert out["value"].iloc[2] == 2.0


def test_align_staleness_cutoff():
    df = _series(["2020-01-01"], [5.0], ["2020-01-01 00:00"])
    dec = pd.Series(pd.DatetimeIndex(["2020-01-05", "2020-03-01"], tz="UTC"),
                    index=pd.DatetimeIndex(["2020-01-05", "2020-03-01"]))
    out = pit.align(df, dec, max_staleness_days=30)
    assert out["value"].iloc[0] == 5.0 and np.isnan(out["value"].iloc[1])


def test_backfill_does_not_replace_newer_observation():
    # obs for Feb published first; a late-arriving Jan value must not overwrite it.
    df = _series(["2020-02-01", "2020-01-01"], [20.0, 10.0], ["2020-03-01", "2020-03-05"])
    dec = pd.Series(pd.DatetimeIndex(["2020-03-10"], tz="UTC"), index=pd.DatetimeIndex(["2020-03-10"]))
    assert pit.align(df, dec)["value"].iloc[0] == 20.0


def test_vintage_aware_derive_and_align():
    # Jan first printed 100 then revised to 110 later; Feb printed 105 (after Jan's revision? no: before)
    df = pd.DataFrame({
        "series_id": "x", "source": "t", "revised": True,
        "obs_date": pd.DatetimeIndex(["2020-01-01", "2020-02-01", "2020-01-01"]),
        "value": [100.0, 105.0, 110.0],
        "available_at": pd.DatetimeIndex(["2020-02-05", "2020-03-05", "2020-04-05"], tz="UTC"),
    })
    d = pit.derive(df, lambda s: s.diff(), "x|diff")
    # As known on 2020-03-05: Feb - Jan(first print) = 5; after 2020-04-05: Feb - Jan(revised) = -5
    dec = pd.Series(pd.DatetimeIndex(["2020-03-10", "2020-04-10"], tz="UTC"),
                    index=pd.DatetimeIndex(["2020-03-10", "2020-04-10"]))
    vals = pit.align(d, dec)["value"].tolist()
    assert vals == [5.0, -5.0]
    # The level as known at 2020-03-10 is Feb's first print, unaffected by the later Jan revision.
    assert pit.align(df, dec)["value"].tolist() == [105.0, 105.0]


def test_derived_series_inherits_latest_availability():
    df = _series(["2020-01-01", "2020-01-02"], [1.0, 3.0], ["2020-01-10", "2020-01-03"])
    d = pit.derive(df, lambda s: s.diff(), "x|diff")
    # diff at obs 01-02 needs obs 01-01, only available 01-10
    assert d.loc[d["obs_date"] == "2020-01-02", "available_at"].iloc[0] == pd.Timestamp("2020-01-10", tz="UTC")


def test_validate_series_rejects_publication_before_period():
    df = _series(["2020-01-10"], [1.0], ["2020-01-05"])
    with pytest.raises(DataValidationError):
        validate_series(df)


def test_feature_context_combined_requires_both_inputs_published(cfg):
    from marketpred.features.base import FeatureContext

    dates = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"])
    dec = decision_times(dates, "NYSE")
    a = _series(["2020-01-02"], [2.0], ["2020-01-02 20:00"]).assign(series_id="A")
    b = _series(["2020-01-02"], [1.0], ["2020-01-03 22:00"]).assign(series_id="B")  # after the 01-03 close
    ctx = FeatureContext(cfg=cfg, prices=pd.DataFrame(index=dates), decision_ts=dec,
                         series=pd.concat([a, b], ignore_index=True))
    out = ctx.combined("A", "B", lambda x, y: x / y, "ratio")
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1]) and out.iloc[2] == 2.0
