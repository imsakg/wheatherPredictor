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
    return forecast_observations(load_observations(args.csv),selected_model(args),args.horizon)


def backtest_data(args):
    from .backtest import rolling_backtest
    rows = load_observations(args.csv)
    return rolling_backtest([row.temperature for row in rows],lambda:selected_model(args),
                            args.min_train,args.horizon,args.step)


def collect_current(args):
    from .api import OpenWeatherClient
    from .collection import save_csv_cache
    item = OpenWeatherClient().current(args.lat,args.lon)
    count = save_csv_cache(args.output,[item],conflict="last")
    return {"count":count,"latest":item.timestamp.isoformat(),"path":args.output}
