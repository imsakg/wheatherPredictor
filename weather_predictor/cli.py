"""Small command-line interface. Data is never fetched or trained on import."""
import argparse
import json
from .csvio import read_csv
from .quality import quality_report


def load_observations(path):
    with open(path,encoding="utf-8",newline="") as stream:
        return read_csv(stream)


def inspect_data(args):
    return quality_report(load_observations(args.csv))


def build_parser():
    parser = argparse.ArgumentParser(prog="weather-predictor")
    commands = parser.add_subparsers(dest="command",required=True)
    inspect = commands.add_parser("inspect",help="report canonical CSV data quality")
    inspect.add_argument("csv")
    inspect.set_defaults(handler=inspect_data)
    from .forecast import MODEL_NAMES
    forecast = commands.add_parser("forecast",help="forecast temperature from canonical CSV")
    add_model_arguments(forecast,MODEL_NAMES)
    forecast.add_argument("--horizon",type=int,default=6)
    forecast.add_argument("--output",help="optional forecast CSV path")
    forecast.set_defaults(handler=forecast_data)
    backtest = commands.add_parser("backtest",help="rolling-origin forecast evaluation")
    add_model_arguments(backtest,MODEL_NAMES)
    backtest.add_argument("--min-train",type=int,default=48)
    backtest.add_argument("--horizon",type=int,default=6)
    backtest.add_argument("--step",type=int,default=6)
    backtest.set_defaults(handler=backtest_data)
    collect = commands.add_parser("collect",help="append current API observation to one-station CSV cache")
    collect.add_argument("--lat",type=float,required=True)
    collect.add_argument("--lon",type=float,required=True)
    collect.add_argument("--output",required=True)
    collect.set_defaults(handler=collect_current)
    report = commands.add_parser("report",help="write a standalone HTML forecast report")
    add_model_arguments(report,MODEL_NAMES)
    report.add_argument("--horizon",type=int,default=6)
    report.add_argument("--output",required=True)
    report.set_defaults(handler=report_data)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0
    except (OSError,ValueError,RuntimeError) as exc:
        parser.error(str(exc))


def add_model_arguments(parser, names):
    parser.add_argument("csv")
    parser.add_argument("--model",choices=names,default="persistence")
    parser.add_argument("--period",type=int,default=24)
    parser.add_argument("--window",type=int,default=6)
    parser.add_argument("--lags",type=int,default=6)


def selected_model(args):
    from .forecast import make_model
    return make_model(args.model,period=args.period,window=args.window,lags=args.lags)


def forecast_data(args):
    from .forecast import forecast_observations
    result = forecast_observations(load_observations(args.csv),selected_model(args),args.horizon)
    if args.output:
        from .export import write_forecast
        with open(args.output,"w",encoding="utf-8",newline="") as stream:
            write_forecast(result,stream)
    return result


def backtest_data(args):
    from .backtest import rolling_backtest
    from .forecast import regular_observations
    rows,_ = regular_observations(load_observations(args.csv))
    return rolling_backtest([row.temperature for row in rows],lambda:selected_model(args),
                            args.min_train,args.horizon,args.step)


def collect_current(args):
    from .api import OpenWeatherClient
    from .collection import save_csv_cache
    item = OpenWeatherClient().current(args.lat,args.lon)
    count = save_csv_cache(args.output,[item],conflict="last")
    return {"count":count,"latest":item.timestamp.isoformat(),"path":args.output}


def report_data(args):
    from .forecast import forecast_observations
    from .report import forecast_report
    rows = load_observations(args.csv)
    predicted = forecast_observations(rows,selected_model(args),args.horizon)
    with open(args.output,"w",encoding="utf-8") as stream:
        stream.write(forecast_report(rows,predicted))
    return {"path":args.output,"horizon":len(predicted),"model":args.model}
