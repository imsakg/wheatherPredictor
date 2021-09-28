"""Forecast errors in original units; sMAPE is percent and zero-zero contributes 0."""
import math
from statistics import mean
from .units import finite


def score(actual, predicted):
    actual, predicted = [finite(v) for v in actual], [finite(v) for v in predicted]
    if not actual or len(actual) != len(predicted):
        raise ValueError("actual and predicted must be nonempty and equally sized")
    errors = [p-a for a,p in zip(actual,predicted)]
    smape = [200*abs(p-a)/(abs(a)+abs(p)) if abs(a)+abs(p) else 0 for a,p in zip(actual,predicted)]
    return {"mae":mean(abs(e) for e in errors), "rmse":math.sqrt(mean(e*e for e in errors)),
            "bias":mean(errors), "smape":mean(smape)}
