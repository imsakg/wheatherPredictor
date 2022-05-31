# Weather Forecaster

A small, dependency-free Python toolkit for collecting station weather and comparing interpretable temperature forecasts. Repository name `wheatherPredictor` is retained; the Python import is `weather_predictor`.

## Quick start

Python 3.9+ and an IANA timezone database are required. On systems without timezone data, install `tzdata`. No TensorFlow, browser driver or plotting backend is needed for the maintained package.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m weather_predictor --help
python -m unittest discover -s tests -v
```

The checkout includes original station files. Run the complete offline workflow without credentials:

```bash
python -m examples.offline_demo --output .cache/demo
python -m weather_predictor inspect .cache/demo/normalized.csv
python -m weather_predictor forecast .cache/demo/normalized.csv   --model holt --horizon 6 --output .cache/forecast.csv
python -m weather_predictor backtest .cache/demo/normalized.csv   --model persistence --min-train 12 --horizon 3 --step 3
python -m weather_predictor report .cache/demo/normalized.csv   --model holt --horizon 6 --output .cache/forecast.html
```

The demo reads real tracked measurements from `datas/2021-03-25.csv`, normalizes them to UTC hourly observations, evaluates held-out rolling forecasts, and writes CSV, SVG, JSON and a self-contained HTML report. One day's example errors are a smoke check, **not evidence of general forecast skill**. The `examples` module is checkout-only, not included in the installed wheel.

## What is included

- Strict timestamped observations in Celsius, hPa, m/s and mm; original daily-CSV conversion with explicit station timezone.
- Deterministic merging, quality reports, hourly/daily aggregation, bounded gap interpolation and robust spike diagnostics.
- Persistence, seasonal-naive, recursive moving-average, exponential smoothing, damped Holt trend, linear trend, ridge AR and differenced AR.
- Chronological splits, training-fitted scaling, lag windows, cyclical features, monthly climatology and weighted ensembles.
- Expanding-origin backtests, MAE/RMSE/bias/sMAPE, per-lead scores, independent-calibration error bands and JSON model recipes.
- OpenWeather current/three-hour forecast integration; bounded HTTPS retries, no unsafe redirects and credential-safe errors.
- Atomic one-station CSV caches, transactional SQLite upserts, forecast exports and offline SVG/HTML visualization.
- Autocorrelation, trailing statistics, dew point, wind chill, heat index and local threshold diagnostics.

CLI commands: `inspect`, `import-daily`, `forecast`, `backtest`, `collect`, `report`. The installed console script `weather-predictor` is equivalent to `python -m weather_predictor`. Library APIs expose additional controls; see [models](docs/models.md).

## Data collection

Set `OPENWEATHER_API_KEY` through your shell/credential manager; do not put keys in source, CSVs or command-line arguments. Then:

```bash
python -m weather_predictor collect --lat 40.18 --lon 29.06   --output .cache/station.csv
```

The key must have the relevant provider entitlement. Collection is a **single current observation**, not a historical backfill or built-in scheduler. Provider forecasts are available through `OpenWeatherClient.forecast()` and must be stored separately from observations. Tests use injected offline responses; they do not prove live API/account availability. See [API integration](docs/api.md).

To normalize original daily files:

```bash
python -m weather_predictor import-daily datas/2021-03-25.csv   --timezone Europe/Istanbul --hourly --output .cache/station.csv
```

Read the [data contract](docs/data.md) before joining sources, summing precipitation or filling gaps. Forecasting/backtesting reject duplicate and irregular timestamps; horizons and seasonal periods count samples, not inherently hours. Input/output aliases are rejected to protect source data.

## Original experiments

`predictor.py`, `parser.py` and `scraper.py` are retained as **legacy experiments**. Importing them does not launch training, rewriting or a browser. Explicitly executing them still requires their old optional dependencies, and their original numerical/data assumptions have not been rehabilitated. The original LSTM/RNN TODO is not presented as a working maintained neural model. Use the package and offline example instead.

## Limitations and development

These scalar statistical baselines are not physical weather simulation, spatial forecasting or official warnings. They can extrapolate poorly. Check against naive baselines on representative held-out seasons; do not use for aviation, flood, medical or other safety-critical decisions. Error bands do not guarantee coverage. SVG axes show sample index; timestamps are provided in forecast CSV/HTML tables.

Run `python -m unittest discover -s tests -v` before contributing. Tests are offline, use temporary files and exercise CLI, transport retries, malformed inputs, import safety and the real-data demo. The GitHub Actions workflow runs tests, compilation and package-build smoke; it has not been run remotely merely by being added locally.

- [Data and timezones](docs/data.md)
- [Models and honest evaluation](docs/models.md)
- [API configuration and security](docs/api.md)
- [Reconstructed history disclosure](HISTORY.md)

License: existing [GNU GPL v3](LICENSE).
