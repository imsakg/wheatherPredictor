# Forecasting and evaluation

These are small, interpretable scalar temperature models, not numerical weather prediction. Humidity, pressure and wind are collected, validated and available for analysis; the supplied forecast models use temperature history only. No neural-network accuracy is claimed for the original TensorFlow experiment.

## Models

- `persistence`: repeats the last observed temperature. Always include this baseline.
- `seasonal`: repeats the last complete period (`SeasonalNaive(period=24)`). At least one full period is required.
- `average`: recursively averages a rolling window. This is **not** the moving-average component of ARIMA.
- `ses`: simple exponential smoothing; the first observation initializes the level. Alpha is explicit in the Python API.
- `holt`: level plus an additive damped trend. Alpha, beta and damping phi are configurable in Python. Damping reduces but does not eliminate extrapolation risk.
- `trend`: ordinary least-squares trend against sample index; needs at least two samples.
- `ar`: ridge autoregression on standardized lags, with an unpenalized intercept. Forecasts recursively feed their own output into later steps.
- `differenced-ar`: AR on first differences, then integration from the final observed temperature. This is an ARIMA-like AR(p,1,0) baseline, not a full ARIMA implementation or automatic order selection.

`MonthlyClimatology` is a separate timestamp-aware monthly average with a training-wide fallback for unseen months. `WeightedEnsemble` creates independent models from factories and averages their forecasts with explicitly normalized nonnegative weights. Select parameters/weights using training and validation data, never final test errors.

AR models use small dense normal equations with partial-pivot elimination. Positive ridge, standardization and explicit lag/workload limits reduce numerical and resource risks; they are not a replacement for a production numerical linear-algebra library. Lags are limited to 128 and the training-size-times-squared-design-width workload to 20 million. Constant series are supported; explosive recursive output can still occur with a poor fit. Inspect forecasts rather than assuming physical plausibility.

## Leakage controls

1. Validate timestamp uniqueness and cadence; aggregate or resolve conflicts explicitly.
2. Split in time, never shuffle samples. `chronological_split` returns three nonempty chronological partitions.
3. Fit scaling on the training partition. Lag-window labels begin strictly after their input window.
4. Tune only on validation folds. Reserve an untouched final test period.
5. For walk-forward evaluation, `rolling_backtest` constructs a **fresh model per origin** and fits only the prefix preceding that origin.

The low-level rolling backtest accepts a scalar series, so the caller must validate original timestamps. The CLI does this automatically. Multi-step folds may overlap when `step < horizon`; aggregate scores then count repeated target timestamps once per origin/lead. Use a non-overlapping step or analyze per-lead results when appropriate. Forecast horizon means samples, not inherently hours.

## Errors and uncertainty

`score` returns MAE, RMSE, signed bias (`predicted - actual`) and sMAPE percent. Zero/zero contributes zero to sMAPE; temperatures near zero can make percentage metrics misleading. MAE/RMSE remain in Celsius.

`residual_intervals` uses the ceiling-ranked finite-sample quantile of absolute **independent calibration** errors and makes symmetric bands. For a requested rank beyond available samples it uses the maximum calibration error. These are empirical bands, not guaranteed coverage under correlated weather, changing regimes or different forecast leads. Calibrate per lead and evaluate actual held-out coverage before operational use. Do not calibrate from the same errors used to claim final model accuracy.

`autocorrelation` uses a global-mean, biased autocovariance normalization. A constant series has lag-zero correlation 1 and other lags 0 by convention. `rolling_statistics` is trailing-only, uses complete windows and population standard deviation.

## Derived indicators

`dew_point` uses a Magnus approximation. `wind_chill` uses the conventional Celsius/km/h formula only at <=10 C and wind above 4.8 km/h. `heat_index` implements the NOAA/Rothfusz calculation including its initial apparent-temperature screen and humidity adjustments. None includes solar exposure, individual health effects, altitude or a warning service. `threshold_alerts` is a local rule diagnostic, not an official weather alert.

References: [Forecasting: Principles and Practice](https://otexts.com/fpp3/), [NOAA heat-index equation](https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml).

## Persistence

`dump_model` writes a versioned JSON recipe containing type, parameters and training temperatures; `load_model` allowlists supported classes and refits deterministically. It does not execute pickle or arbitrary imports. The format is for small experiments: 2 MB documents, at most 20,000 training samples on load, plus AR-specific limits. Model recipes contain station history: treat them as data, not credentials. Ensembles and monthly climatology are not currently serializable through this recipe format.
