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


def forecast_observations(observations, model, horizon=6, step=timedelta(hours=1)):
    rows = sorted(observations,key=lambda row:row.timestamp)
    if not rows:
        raise ValueError("observations must not be empty")
    if step.total_seconds() <= 0:
        raise ValueError("step must be positive")
    predictions = model.fit([row.temperature for row in rows]).predict(horizon)
    return [{"timestamp":(rows[-1].timestamp+step*(i+1)).isoformat(),"temperature":value}
            for i,value in enumerate(predictions)]
