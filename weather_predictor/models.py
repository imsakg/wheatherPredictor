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


class ExponentialSmoothing(ForecastModel):
    def __init__(self, alpha=0.3):
        self.alpha = finite(alpha)
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0,1]")

    def fit(self, values):
        self.history = _series(values)
        self.level = self.history[0]
        for value in self.history[1:]:
            self.level = self.alpha*value + (1-self.alpha)*self.level
        return self

    def predict(self, horizon):
        self._check(horizon)
        return [self.level]*horizon


class HoltTrend(ExponentialSmoothing):
    """Holt's level and damped additive trend."""
    def __init__(self, alpha=0.3, beta=0.1, phi=0.98):
        super().__init__(alpha)
        self.beta, self.phi = finite(beta), finite(phi)
        if not 0 <= self.beta <= 1 or not 0 < self.phi <= 1:
            raise ValueError("beta must be in [0,1] and phi in (0,1]")

    def fit(self, values):
        self.history = _series(values,2)
        self.level = self.history[0]
        self.trend = self.history[1]-self.history[0]
        for value in self.history[1:]:
            previous = self.level
            self.level = self.alpha*value + (1-self.alpha)*(self.level+self.phi*self.trend)
            self.trend = self.beta*(self.level-previous) + (1-self.beta)*self.phi*self.trend
        return self

    def predict(self, horizon):
        self._check(horizon)
        cumulative = 0.0
        result = []
        for h in range(1,horizon+1):
            cumulative += self.phi**h
            result.append(self.level+self.trend*cumulative)
        return result


class LinearTrend(ForecastModel):
    def fit(self, values):
        self.history = _series(values,2)
        n = len(self.history)
        xbar, ybar = (n-1)/2, mean(self.history)
        self.slope = sum((i-xbar)*(y-ybar) for i,y in enumerate(self.history))/sum((i-xbar)**2 for i in range(n))
        self.intercept = ybar-self.slope*xbar
        return self

    def predict(self, horizon):
        self._check(horizon)
        return [self.intercept+self.slope*(len(self.history)+h) for h in range(horizon)]
