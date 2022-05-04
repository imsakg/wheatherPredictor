"""Run with python -m examples.offline_demo; uses real tracked station measurements."""
import argparse
import io
import json
from datetime import date
from pathlib import Path
from weather_predictor.legacy import read_daily
from weather_predictor.preprocessing import hourly_mean
from weather_predictor.forecast import forecast_observations
from weather_predictor.models import HoltTrend
from weather_predictor.backtest import rolling_backtest
from weather_predictor.csvio import write_csv
from weather_predictor.export import write_forecast
from weather_predictor.visualization import forecast_chart, residual_chart
from weather_predictor.report import forecast_report
from weather_predictor.output import atomic_text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default=".cache/demo")
    args = parser.parse_args(argv)
    target = Path(args.output)
    source = Path(__file__).resolve().parents[1]/"datas"/"2021-03-25.csv"
    with source.open(encoding="utf-8",newline="") as stream:
        rows = hourly_mean(read_daily(stream,date(2021,3,25),"Europe/Istanbul"))
    predicted = forecast_observations(rows,HoltTrend(),6)
    evaluation = rolling_backtest([row.temperature for row in rows],HoltTrend,12,3,3)
    stream = io.StringIO(); write_csv(rows,stream)
    atomic_text(target/"normalized.csv",stream.getvalue())
    stream = io.StringIO(); write_forecast(predicted,stream)
    atomic_text(target/"forecast.csv",stream.getvalue())
    atomic_text(target/"forecast.svg",forecast_chart([row.temperature for row in rows],[row["temperature"] for row in predicted]))
    records = evaluation["records"]
    atomic_text(target/"residuals.svg",residual_chart([r["actual"] for r in records],[r["predicted"] for r in records]))
    atomic_text(target/"report.html",forecast_report(rows,predicted))
    atomic_text(target/"backtest.json",json.dumps(evaluation,indent=2,allow_nan=False))
    print(json.dumps({"observations":len(rows),"folds":evaluation["folds"],"metrics":evaluation["metrics"],"output":str(target)},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
