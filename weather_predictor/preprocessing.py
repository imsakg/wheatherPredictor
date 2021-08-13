"""UTC aggregation with circular wind means; empty bins are omitted."""
import math
from collections import defaultdict
from statistics import mean
from .schema import Observation
from .csvio import COLUMNS


def hourly_mean(observations):
    groups = defaultdict(list)
    for item in observations:
        groups[item.timestamp.replace(minute=0, second=0, microsecond=0)].append(item)
    result = []
    for stamp, rows in sorted(groups.items()):
        values = {}
        for name in COLUMNS[1:]:
            data = [getattr(row,name) for row in rows if getattr(row,name) is not None]
            if not data:
                values[name] = None
            elif name == "wind_direction":
                x = sum(math.cos(math.radians(v)) for v in data)
                y = sum(math.sin(math.radians(v)) for v in data)
                values[name] = None if math.hypot(x,y) < 1e-12 else math.degrees(math.atan2(y,x)) % 360
                if values[name] is not None and values[name] > 360 - 1e-10:
                    values[name] = 0.0
            else:
                values[name] = sum(data) if name == "precipitation" else mean(data)
        result.append(Observation(stamp, **values))
    return result


def interpolate_short_gaps(values, max_gap=2):
    """Linearly fill bounded scalar gaps; use only within a training partition."""
    if not isinstance(max_gap,int) or max_gap < 0:
        raise ValueError("max_gap must be a nonnegative integer")
    result = list(values)
    i = 0
    while i < len(result):
        if result[i] is not None:
            i += 1
            continue
        start = i
        while i < len(result) and result[i] is None:
            i += 1
        length = i-start
        if start > 0 and i < len(result) and length <= max_gap:
            left, right = result[start-1], result[i]
            for offset in range(length):
                result[start+offset] = left + (right-left)*(offset+1)/(length+1)
    return result
