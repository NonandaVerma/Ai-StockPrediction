-- 001: core schema for the Astrology Market Forecast section.
-- Raw source data (price_bars, planetary_positions) is stored separately from derived data
-- (astro_events, predictions, backtest_runs) so results can be reproduced and recalculated.
-- All timestamps are ISO-8601 UTC strings ("YYYY-MM-DDTHH:MM:SS+00:00").

CREATE TABLE data_sources (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,          -- e.g. "yahoo", "alphavantage", "swisseph-moshier"
    kind          TEXT NOT NULL,                 -- market | ephemeris
    realtime      INTEGER NOT NULL DEFAULT 0,    -- 1 only if the provider contractually delivers real-time data
    notes         TEXT,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S+00:00','now'))
);

CREATE TABLE instruments (
    id            INTEGER PRIMARY KEY,
    symbol        TEXT NOT NULL UNIQUE,          -- provider-neutral symbol as entered (e.g. "^NSEI", "RELIANCE.NS")
    name          TEXT,
    exchange      TEXT NOT NULL,                 -- registry code: NSE, BSE, NYSE, NASDAQ, LSE
    sector        TEXT,
    currency      TEXT,
    kind          TEXT,                          -- index | equity | etf | other
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S+00:00','now')),
    updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S+00:00','now'))
);

CREATE TABLE price_bars (
    id            INTEGER PRIMARY KEY,
    instrument_id INTEGER NOT NULL REFERENCES instruments(id),
    source_id     INTEGER NOT NULL REFERENCES data_sources(id),
    interval      TEXT NOT NULL,                 -- 1d | 1h | 15m | 5m
    session_date  TEXT NOT NULL,                 -- exchange-local trading date (YYYY-MM-DD)
    ts_utc        TEXT NOT NULL,                 -- bar start (intraday) or session close (daily), UTC
    open REAL, high REAL, low REAL,
    close         REAL NOT NULL,
    adj_close     REAL,                          -- split/dividend adjusted close (NULL if provider has none)
    volume        REAL,
    fetched_at    TEXT NOT NULL,
    UNIQUE (instrument_id, interval, ts_utc)
);
CREATE INDEX ix_price_bars_lookup ON price_bars (instrument_id, interval, session_date);

CREATE TABLE data_freshness (
    instrument_id INTEGER NOT NULL REFERENCES instruments(id),
    interval      TEXT NOT NULL,
    source_id     INTEGER NOT NULL REFERENCES data_sources(id),
    first_bar_utc TEXT,
    last_bar_utc  TEXT,
    fetched_at    TEXT NOT NULL,
    label         TEXT NOT NULL,                 -- real-time | delayed | end-of-day | synthetic
    PRIMARY KEY (instrument_id, interval, source_id)
);

-- Astronomical facts (no interpretation): one row per body per instant per frame.
CREATE TABLE planetary_positions (
    id            INTEGER PRIMARY KEY,
    ts_utc        TEXT NOT NULL,
    zodiac        TEXT NOT NULL,                 -- tropical | sidereal
    ayanamsa      TEXT NOT NULL,                 -- "none" for tropical
    body          TEXT NOT NULL,
    longitude     REAL NOT NULL,                 -- ecliptic longitude, degrees
    latitude      REAL,
    distance_au   REAL,
    speed_deg_day REAL,
    declination   REAL,
    ephemeris     TEXT NOT NULL,                 -- swieph | moshier
    UNIQUE (ts_utc, zodiac, ayanamsa, body, ephemeris)
);
CREATE INDEX ix_positions_ts ON planetary_positions (ts_utc);

-- Astrological interpretation layer: derived events (ingress, stations, lunations, aspects).
CREATE TABLE astro_events (
    id            INTEGER PRIMARY KEY,
    event_date    TEXT NOT NULL,                 -- UTC date on which the event is first detected
    event_type    TEXT NOT NULL,
    body1         TEXT,
    body2         TEXT,
    detail        TEXT,
    zodiac        TEXT NOT NULL,
    ayanamsa      TEXT NOT NULL,
    UNIQUE (event_date, event_type, body1, body2, zodiac, ayanamsa)
);
CREATE INDEX ix_events_date ON astro_events (event_date);

CREATE TABLE model_versions (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    version       TEXT NOT NULL,                 -- code version + parameter hash
    family        TEXT NOT NULL,                 -- baseline | market | astro | hybrid
    params_json   TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S+00:00','now')),
    UNIQUE (name, version)
);

CREATE TABLE backtest_runs (
    id               INTEGER PRIMARY KEY,
    run_key          TEXT NOT NULL UNIQUE,       -- hash of inputs; identical reruns are not duplicated
    instrument_id    INTEGER NOT NULL REFERENCES instruments(id),
    model_version_id INTEGER NOT NULL REFERENCES model_versions(id),
    gap_sessions     INTEGER NOT NULL,
    horizon_sessions INTEGER NOT NULL,
    zodiac           TEXT NOT NULL,
    ayanamsa         TEXT NOT NULL,
    data_start       TEXT NOT NULL,
    data_end         TEXT NOT NULL,
    eval_start       TEXT NOT NULL,
    eval_end         TEXT NOT NULL,
    folds            INTEGER NOT NULL,
    cost_bps         REAL NOT NULL,
    metrics_json     TEXT NOT NULL,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S+00:00','now'))
);
CREATE INDEX ix_backtest_lookup ON backtest_runs (instrument_id, horizon_sessions, created_at);

CREATE TABLE predictions (
    id                INTEGER PRIMARY KEY,
    instrument_id     INTEGER NOT NULL REFERENCES instruments(id),
    model_version_id  INTEGER NOT NULL REFERENCES model_versions(id),
    as_of_session     TEXT NOT NULL,             -- last session whose close the forecast could see
    window_start      TEXT NOT NULL,             -- first trading session of the target window
    window_end        TEXT NOT NULL,             -- last trading session of the target window
    gap_sessions      INTEGER NOT NULL,
    horizon_sessions  INTEGER NOT NULL,
    zodiac            TEXT NOT NULL,
    ayanamsa          TEXT NOT NULL,
    direction         TEXT,                      -- bullish | bearish | neutral | NULL (unavailable)
    prob_up           REAL,
    expected_return   REAL,
    interval_low      REAL,
    interval_high     REAL,
    expected_vol      REAL,
    risk_category     TEXT,
    confidence        REAL,                      -- validated hit rate; NULL if not calibrated
    evidence          TEXT NOT NULL,             -- validated | experimental | insufficient
    details_json      TEXT,
    generated_at      TEXT NOT NULL,
    realized_return   REAL,                      -- filled in once the window has closed
    UNIQUE (instrument_id, model_version_id, as_of_session, window_start, window_end, zodiac, ayanamsa)
);
CREATE INDEX ix_predictions_lookup ON predictions (instrument_id, as_of_session);
