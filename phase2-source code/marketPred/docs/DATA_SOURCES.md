# Data sources

All sources below were accessed keylessly from the reference environment on 2026-10-16. "Observed
coverage" is what ingestion actually stored (see `data/metadata.db`, table `fetch_log`). Licensing notes
summarise the providers' public terms as I understand them. **Verify the current terms yourself before
any redistribution or commercial use.**

Every series is stored with `obs_date` (reference period) and `available_at` (UTC timestamp from which a
market participant could have known it). `available_at` rules are configured in `configs/default.yaml`.
Rows that are not yet published at a decision time are invisible to the as-of join (`data/pit.py`).

| Source (connector) | Series used | Frequency | Observed coverage | Publication rule (`available_at`) | Revisions | Access / licence |
|---|---|---|---|---|---|---|
| Yahoo Finance chart API (`yahoo`) | `^GSPC` OHLCV (target) | Daily | 1927-; used from 1988-11 | NYSE close (calendar-aware, incl. early closes); unfinished bars dropped | Rare vendor corrections | Unofficial, keyless, may break or rate-limit; Yahoo terms restrict redistribution. Research use only |
| Yahoo (`yahoo_cross_asset`) | `^VIX`, `^VIX3M`, `DX-Y.NYB`, `GC=F`, `CL=F`, `HG=F`, `HYG`, `LQD`, `^RUT` closes | Daily | VIX 1990-, VIX3M 2006-07, DXY 1985-, futures 2000-08, LQD 2002-07, HYG 2007-04, RUT 1987- | Own close time: VIX 16:15 ET, futures/DXY 17:00 ET, ETFs/RUT 16:00 ET | None | As above. Continuous futures (`=F`) are front-month splices with roll jumps |
| US Treasury (`treasury`) | Par yields 3M, 2Y, 10Y, 30Y | Daily | 1990-01-02- (30Y gap 2002-2006) | Same day 18:00 ET (used from next close) | Not revised | Public domain (US government) |
| Chicago Fed NFCI (`nfci`) | NFCI, ANFCI | Weekly (Fri) | 1971- ; **file ended 2026-04-24 when downloaded** | Following Wednesday 08:30 ET | **Whole history re-estimated weekly**; only latest vintage keyless → revision look-ahead (flagged `revised=True`) | Free; FRED/ALFRED hold vintages (key required) |
| U. Michigan Surveys of Consumers (`umich`) | Index of Consumer Sentiment | Monthly | 1952-11 (monthly from 1978) - 2026-09 | Month end + 1 day 10:00 ET | Preliminary vs final not distinguished (final used) | Free for non-commercial use per the Surveys' terms |
| BLS public API v2 (`bls`) | CPI-U NSA (`CUUR0000SA0`), unemployment rate (`LNS14000000`) | Monthly | 1988-01 - 2026-09; Oct-2025 CPI absent (not published) | CPI: month end + 35 d; UNRATE: month end + 55 d (covers 2013 and 2025 shutdown delays) | CPI NSA not revised; UNRATE SA revised annually (small) | Public domain; keyless limit 25 queries/day, 10 years/query (`BLS_API_KEY` raises limits) |
| Shiller online data (`shiller`) | S&P composite earnings E, dividends D | Monthly | 1871 - **2024-06** (file not updated after 2024) | Month end + 120 days | Recent E are estimates later replaced; history occasionally revised | Academic dataset, free; cite R. Shiller |
| Baker-Bloom-Davis EPU (`epu`) | Daily US Economic Policy Uncertainty | Daily | 1985-01-01 - 2026-10-15 | obs + 2 days | Recent days can change as newspapers are re-processed | Free with citation (policyuncertainty.com) |
| Caldara-Iacoviello GPR (`gpr`) | Daily GPR, acts, threats | Daily | 1985-01-01 - 2026-10-13 | obs + 7 days (file updated roughly weekly) | Methodology updates possible | Free with citation |
| CFTC Commitments of Traders (`cftc`) | E-mini S&P 500 (13874A) legacy futures-only positions | Weekly (Tue) | 1997-09-16 - 2026-10-06 | Tuesday + 6 days 09:00 ET (published Friday 15:30 ET; Monday used as holiday buffer) | Occasional corrections | Public domain; Socrata API |
| NASA POWER (`nasa_power`) | NYC cloud amount, clearness index, temperature, precipitation | Daily | 1984- (clearness 2001-); cloud ended 2026-07-30 when downloaded | obs + 1 day | Reanalysis (MERRA-2/GEOS) - **not** real-time observations; near-real-time values are later replaced | Free, NASA open data |
| FRED/ALFRED (`fred`, optional) | e.g. BAA10Y, T10YIE; vintages via `realtime_start` | Varies | not used (keyless endpoints unreachable from this network; API needs `FRED_API_KEY`) | `realtime_start` 23:59 ET | Vintage-aware (`pit.derive`/`pit.align` handle multiple vintages) | Free API key; per-series third-party copyrights apply |

## Sources investigated but not used

| Source | Reason |
|---|---|
| FRED `fredgraph.csv` (keyless) | Timed out / connection reset from this network |
| DBnomics mirror of BLS | Reachable but stale (ended 2025-01); FED H15 mirror was current |
| AAII Sentiment Survey | HTTP 403 for automated download |
| GDELT DOC 2.0 API | Rate-limited (one request / 5 s) and article search covers only a recent rolling window; not suitable for a 1990- backtest |
| Google Trends | Values are sampled and re-normalised per request; cannot be reconstructed point-in-time |
| Swiss Ephemeris | AGPL-3.0 or paid commercial licence; PyEphem (LGPL-3.0) is sufficient for the predefined variables |
| NOAA Climate Data Online | Requires a token; NASA POWER covers the needed variables keylessly |
| Exchange data (NSE/BSE, NYSE TAQ), options (OptionMetrics), fundamentals (Compustat, I/B/E/S), news archives (Dow Jones, Refinitiv, RavenPack) | Paid / licensed; connectors can be added via `data/base.Connector` |

## Known data limitations

1. **^GSPC open prices before ~2007 are unreliable** (41% of opens equal the previous close in the
   ingested history; >90% in 1996-2005). No feature uses the open.
2. **Price index excludes dividends.** Return targets and the trading simulation understate total return.
3. **NFCI and Shiller are latest-vintage** downloads of revised series. NFCI revisions can leak
   information into macro features; Shiller E is lagged 120 days which mitigates but does not remove it.
4. **Stale sources become missing**: values older than `max_staleness_days` are set to NaN instead of
   being carried forward (e.g. valuation features are missing after 2024-12 because Shiller's file ends
   in 2024-06; NFCI is missing after mid-May 2026).
5. **Reanalysis weather is not what traders observed in real time**; with a one-day lag the difference
   is small for the variables used, but it is not a true real-time record.
6. **Negative futures price:** WTI (`CL=F`) settled at -37.63 on 2020-04-20; log returns are undefined
   there, so `oil_ret21` is missing on the two affected dates rather than patched.
7. **NIFTY 50 calendar gaps (extension market):** Yahoo omits 27 NSE sessions (mostly one-hour Diwali
   *Muhurat* sessions) and includes the 2025-02-01 Budget-day Saturday session, which the NSE calendar
   library does not list (dropped). Targets count *available* sessions, so a skipped Muhurat session is
   merged into the next return.
8. **Release-time assumptions are conservative constants**, not historical release calendars. Exact
   historical release dates (e.g. from ALFRED) would make macro features slightly more timely.
