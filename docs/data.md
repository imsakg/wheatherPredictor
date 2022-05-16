# Observation data contract

The maintained package is `weather_predictor`; one CSV cache or SQLite database represents **one station**. There is no spatial weather model and no automatic station mixing.

## Canonical CSV

Required columns are `timestamp,temperature`. Optional columns are `humidity,pressure,wind_speed,wind_direction,precipitation`.

- `timestamp`: ISO 8601 with an explicit offset (`Z` is accepted); observations normalize to UTC.
- `temperature`: finite Celsius, never missing.
- `humidity`: relative humidity percent, 0 through 100; empty means unknown.
- `pressure`: positive hPa; empty means unknown. Distinguish station and sea-level pressure upstream when joining sources.
- `wind_speed`: nonnegative metres/second.
- `wind_direction`: meteorological direction **from which** wind blows, degrees in [0,360); unknown/calm direction is empty.
- `precipitation`: nonnegative liquid-equivalent millimetres **for the source interval**. Missing is unknown, not an inferred zero.

The reader sorts by timestamp but does not delete duplicates. Duplicate headers, extra cells and invalid required cells are errors with row context. `quality_report` reports duplicates, missing fields and cadence gaps. `merge_observations` removes identical duplicates and rejects conflicting duplicates unless `conflict="last"` is explicitly selected.

Forecasting and CLI backtesting require unique, regularly spaced timestamps. Cadence is inferred from the observations, not assumed hourly. Period/window/lag/horizon values count **samples**, so a period of 24 means one day only for hourly data. `inspect` defaults to an expected hourly cadence for its diagnostic missing-step counts; it does not change the data.

## Original tracked files

`datas/YYYY-MM-DD.csv` is the original daily station scrape. It contains Fahrenheit, mph and inches of mercury plus station-local clock strings. Read it through:

```bash
python -m weather_predictor import-daily datas/2021-03-25.csv   --timezone Europe/Istanbul --hourly --output .cache/station.csv
```

The importer handles signed/decimal and three-digit temperatures, midnight/noon, non-breaking-space unit suffixes, 32 compass points, and the original scraper's completely empty indexed padding rows. Malformed nonempty rows fail rather than silently disappearing. The timezone must describe the station at the historical date, not your workstation timezone. Ambiguous or nonexistent daylight-saving clock times must be resolved explicitly upstream; the maintained importer rejects them.

The old `data.csv` and `data/data.csv` are **not** canonical: their legacy converter truncated numbers and used uncertain wind units. Do not rename their columns and assume they are clean. Reimport the original daily files where available. Original files are retained without inventing missing measurements.

## Aggregation and imputation

`hourly_mean` groups in UTC, averages scalar measurements, uses circular direction means, and sums interval precipitation. Opposing winds with an undefined vector mean produce a missing direction. Empty hours are omitted, not invented; a later regular-cadence check will catch gaps. Only sum precipitation if input rows represent **non-overlapping interval amounts**. Snapshot rolling-hour totals or cumulative gauge readings need source-specific differencing/deduplication first; this library cannot infer that metadata.

`daily_summary` groups by an explicit IANA timezone and reports observed precipitation subtotals and their sample counts. A partial day is not a full-day rainfall estimate.

`interpolate_short_gaps` fills only bounded scalar gaps up to the configured limit. Partition train/validation/test **before** interpolation. Its use of a right endpoint is not causal online inference. Hampel outlier flags are centered retrospective diagnostics, not automatic deletion or live predictors.

CSV replacement is atomic but assumes one writer per cache; use `ObservationStore` for transactional upserts. SQLite preserves a source label, but does not enforce units or station identity beyond canonical `Observation` validation. Keep provider forecasts, local forecasts and observed measurements in separate files/stores.
