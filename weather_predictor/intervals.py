"""Symmetric empirical error bands, not guaranteed meteorological confidence levels."""
import math
from .models import _series
from .units import finite


def residual_intervals(predicted, calibration_residuals, coverage=0.9):
    coverage = finite(coverage)
    if not 0 < coverage < 1:
        raise ValueError("coverage must be in (0,1)")
    errors = sorted(abs(v) for v in _series(calibration_residuals))
    rank = min(len(errors), math.ceil((len(errors)+1)*coverage))
    radius = errors[rank-1]
    return [(value-radius,value+radius) for value in _series(predicted)]
