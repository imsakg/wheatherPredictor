"""Dependency-free scalar forecasting models; all predictions are out-of-sample."""
from statistics import mean
from .units import finite


def _series(values, minimum=1):
    result = [finite(value) for value in values]
    if len(result) < minimum:
        raise ValueError("need at least {} training samples".format(minimum))
    return result


class ForecastModel:
    def _check(self, horizon):
        if not hasattr(self,"history"):
            raise ValueError("model is not fitted")
        if not isinstance(horizon,int) or isinstance(horizon,bool) or horizon < 1:
            raise ValueError("horizon must be a positive integer")


class Persistence(ForecastModel):
    def fit(self, values):
        self.history = _series(values)
        return self

    def predict(self, horizon):
        self._check(horizon)
        return [self.history[-1]]*horizon
