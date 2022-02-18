"""Forecast CSV exports are separate from observations and preserve optional bands."""
import csv
from datetime import datetime
from .units import finite


def write_forecast(predictions, stream, intervals=None):
    rows = list(predictions)
    if intervals is not None:
        intervals = list(intervals)
        if len(rows) != len(intervals):
            raise ValueError("intervals must match forecast length")
    validated = []
    for i,row in enumerate(rows):
        stamp = datetime.fromisoformat(row["timestamp"].replace("Z","+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("forecast timestamps must include a timezone")
        item = {"timestamp":stamp.isoformat(),"temperature":finite(row["temperature"])}
        if intervals is not None:
            lower, upper = map(finite,intervals[i])
            if lower > item["temperature"] or upper < item["temperature"]:
                raise ValueError("interval must contain its prediction")
            item.update(lower=lower,upper=upper)
        validated.append(item)
    columns = ["timestamp","temperature"]+(["lower","upper"] if intervals is not None else [])
    writer = csv.DictWriter(stream,fieldnames=columns,lineterminator="\n")
    writer.writeheader()
    writer.writerows(validated)
