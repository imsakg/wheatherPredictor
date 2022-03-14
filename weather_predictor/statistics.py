"""Small descriptive tools for inspecting seasonality and recent weather regimes."""
import math
from statistics import mean
from .models import _series


def autocorrelation(values, max_lag=24):
    data = _series(values)
    if not isinstance(max_lag,int) or not 0 <= max_lag < len(data):
        raise ValueError("max_lag must be nonnegative and shorter than the series")
    centered = [value-mean(data) for value in data]
    denominator = sum(value*value for value in centered)
    return [1.0]+[sum(a*b for a,b in zip(centered[:-lag],centered[lag:]))/denominator
                  if denominator else 0.0 for lag in range(1,max_lag+1)]


def rolling_statistics(values, window=24):
    data = _series(values)
    if not isinstance(window,int) or not 1 <= window <= len(data):
        raise ValueError("window must be a positive integer no longer than the series")
    output = []
    for end in range(window,len(data)+1):
        local = data[end-window:end]
        center = mean(local)
        output.append({"index":end-1,"mean":center,"min":min(local),"max":max(local),
                       "std":math.sqrt(mean((v-center)**2 for v in local))})
    return output
