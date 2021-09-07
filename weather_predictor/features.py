"""Cyclical calendar and meteorological (wind-from) vector features."""
import math
from datetime import datetime, timezone
from .units import finite


def calendar_features(stamp):
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    stamp = stamp.astimezone(timezone.utc)
    day_phase = (stamp.hour*3600+stamp.minute*60+stamp.second)/86400
    year_start = datetime(stamp.year,1,1,tzinfo=timezone.utc)
    year_end = datetime(stamp.year+1,1,1,tzinfo=timezone.utc)
    year_phase = (stamp-year_start)/(year_end-year_start)
    return {"hour_sin":math.sin(2*math.pi*day_phase), "hour_cos":math.cos(2*math.pi*day_phase),
            "year_sin":math.sin(2*math.pi*year_phase), "year_cos":math.cos(2*math.pi*year_phase)}


def wind_components(speed, direction):
    speed, direction = finite(speed), finite(direction)
    if speed < 0 or not 0 <= direction < 360:
        raise ValueError("wind requires speed >= 0 and direction in [0,360)")
    angle = math.radians(direction)
    return -speed*math.sin(angle), -speed*math.cos(angle)
