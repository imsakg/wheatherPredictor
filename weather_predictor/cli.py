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
