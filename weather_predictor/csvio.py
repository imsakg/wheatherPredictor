"""Strict, portable CSV import/export without dataframe dependencies."""
import csv
from dataclasses import fields
from datetime import datetime
from .schema import Observation

COLUMNS = [field.name for field in fields(Observation)]


def read_csv(stream):
    reader = csv.DictReader(stream)
    if not {"timestamp", "temperature"}.issubset(reader.fieldnames or []):
        raise ValueError("CSV requires timestamp and temperature columns")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("duplicate CSV columns")
    result = []
    for row in reader:
        try:
            if None in row or not row.get("timestamp"):
                raise ValueError("row has extra cells or missing timestamp")
            values = {name: float(row[name]) if row.get(name) not in (None, "") else None
                      for name in COLUMNS[1:]}
            if values["temperature"] is None:
                raise ValueError("temperature is required")
            stamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            result.append(Observation(stamp, **values))
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("line {}: {}".format(reader.line_num, exc)) from exc
    return sorted(result, key=lambda row: row.timestamp)


def write_csv(observations, stream):
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for item in observations:
        row = {name: getattr(item, name) for name in COLUMNS}
        row["timestamp"] = item.timestamp.isoformat()
        writer.writerow(row)
