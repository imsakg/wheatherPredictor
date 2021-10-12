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


class SeasonalNaive(ForecastModel):
    def __init__(self, period=24):
        if not isinstance(period,int) or period < 1:
            raise ValueError("period must be a positive integer")
        self.period = period

    def fit(self, values):
        self.history = _series(values,self.period)
        return self

    def predict(self, horizon):
        self._check(horizon)
        season = self.history[-self.period:]
        return [season[i % self.period] for i in range(horizon)]


class MovingAverage(SeasonalNaive):
    """Recursive forecast of the recent sample mean, not an ARIMA MA term."""
    def __init__(self, window=6):
        super().__init__(window)
        self.window = window

    def predict(self, horizon):
        self._check(horizon)
        buffer = self.history[-self.window:]
        predictions = []
        for _ in range(horizon):
            predictions.append(mean(buffer[-self.window:]))
            buffer.append(predictions[-1])
        return predictions
