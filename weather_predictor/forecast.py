"""Model registry and timestamped, temperature-only local forecasting."""
from datetime import timedelta
from .models import Persistence, SeasonalNaive, MovingAverage, ExponentialSmoothing, HoltTrend, LinearTrend, AutoRegression, DifferencedAR

MODEL_NAMES = ("persistence","seasonal","average","ses","holt","trend","ar","differenced-ar")


def make_model(name, period=24, window=6, lags=6):
    factories = {"persistence":Persistence,"seasonal":lambda:SeasonalNaive(period),
                 "average":lambda:MovingAverage(window),"ses":ExponentialSmoothing,
                 "holt":HoltTrend,"trend":LinearTrend,"ar":lambda:AutoRegression(lags),
                 "differenced-ar":lambda:DifferencedAR(lags)}
    if name not in factories:
        raise ValueError("unknown forecasting model")
    return factories[name]()


def regular_observations(observations, step=None):
    rows = sorted(observations,key=lambda row:row.timestamp)
    if not rows or (len(rows)<2 and step is None):
        raise ValueError("need at least two observations to infer cadence, or supply an explicit step")
    if step is None:
        step = rows[1].timestamp-rows[0].timestamp
    if step.total_seconds() <= 0:
        raise ValueError("timestamps must be unique and cadence positive")
    if any(b.timestamp-a.timestamp != step for a,b in zip(rows,rows[1:])):
        raise ValueError("timestamps must be unique and regularly spaced; resample first")
    return rows,step


def forecast_observations(observations, model, horizon=6, step=None):
    rows,step = regular_observations(observations,step)
    predictions = model.fit([row.temperature for row in rows]).predict(horizon)
    return [{"timestamp":(rows[-1].timestamp+step*(i+1)).isoformat(),"temperature":value}
            for i,value in enumerate(predictions)]
