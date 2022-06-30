"""Portable, allowlisted JSON model recipes. Never unpickle downloaded models."""
import json
from . import models

_PARAMETERS = {
    "Persistence": (), "SeasonalNaive": ("period",), "MovingAverage": ("window",),
    "ExponentialSmoothing": ("alpha",), "HoltTrend": ("alpha","beta","phi"),
    "LinearTrend": (), "AutoRegression": ("lags","ridge"), "DifferencedAR": ("lags","ridge"),
}


def dump_model(model, stream):
    name = type(model).__name__
    if name not in _PARAMETERS or not hasattr(model,"history"):
        raise ValueError("unsupported or unfitted model")
    payload = {"version":1, "model":name, "parameters":{k:getattr(model,k) for k in _PARAMETERS[name]},
               "training":getattr(model,"original_history",model.history)}
    json.dump(payload,stream,allow_nan=False,indent=2)


def load_model(stream):
    try:
        text = stream.read(2000001)
        if len(text) > 2000000:
            raise ValueError("model document exceeds 2 MB limit")
        payload = json.loads(text)
        if not isinstance(payload,dict):
            raise ValueError("model document must be an object")
        name = payload["model"]
        if type(payload["version"]) is not int or payload["version"] != 1 or name not in _PARAMETERS:
            raise ValueError("unsupported model format")
        params, training = payload["parameters"], payload["training"]
        if set(params) != set(_PARAMETERS[name]) or not isinstance(training,list) or len(training)>20000:
            raise ValueError("invalid model parameters or training data")
        return getattr(models,name)(**params).fit(training)
    except (KeyError, TypeError, json.JSONDecodeError, OverflowError) as exc:
        raise ValueError("invalid model document") from exc
