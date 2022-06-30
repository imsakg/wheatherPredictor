"""Read original Wunderground daily files, correcting the old sliced-number parser."""
import csv
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from .schema import Observation
from .units import fahrenheit_to_celsius, mph_to_ms, inhg_to_hpa, inches_to_mm

_POINTS = "N NBE NNE NEBN NE NEBE ENE EBN E EBS ESE SEBE SE SEBS SSE SBE S SBW SSW SWBS SW SWBW WSW WBS W WBN WNW NWBW NW NWBN NNW NBW".split()


def compass(value):
    value = value.strip().upper()
    if value in ("CALM", "VAR", "", "--"):
        return None
    if value not in _POINTS:
        raise ValueError("unknown wind direction: " + value)
    return _POINTS.index(value) * 11.25


def number(value):
    match = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)", value)
    if not match:
        raise ValueError("invalid measurement: " + value)
    return float(match.group(1))


def read_daily(stream, day, timezone_name):
    try:
        zone = ZoneInfo(timezone_name)
    except KeyError:
        raise ValueError("unknown station timezone") from None
    result = []
    reader = csv.DictReader(stream)
    for row in reader:
        # Original scraper emitted indexed padding rows with every measurement empty.
        if all(not row.get(name) for name in ("Time","Temperature","Humidity","Wind","Wind Speed","Pressure","Precip.","Condition")):
            continue
        try:
            clock = datetime.strptime(row["Time"].strip(), "%I:%M %p").time()
            naive = datetime.combine(day,clock)
            stamp = naive.replace(tzinfo=zone)
            if stamp.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) != naive:
                raise ValueError("nonexistent station local clock; resolve offset explicitly")
            if stamp.utcoffset() != naive.replace(tzinfo=zone,fold=1).utcoffset():
                raise ValueError("ambiguous station local clock; resolve offset explicitly")
            rain = row.get("Precip.", "")
            result.append(Observation(
                stamp, fahrenheit_to_celsius(number(row["Temperature"])),
                humidity=number(row["Humidity"]), pressure=inhg_to_hpa(number(row["Pressure"])),
                wind_speed=mph_to_ms(number(row["Wind Speed"])),
                wind_direction=compass(row["Wind"]),
                precipitation=inches_to_mm(number(rain)) if rain else None))
        except (KeyError, ValueError) as exc:
            raise ValueError("line {}: {}".format(reader.line_num, exc)) from exc
    return sorted(result, key=lambda row: row.timestamp)
