"""Expanding-window, rolling-origin evaluation. Every fold fits a fresh model."""
from .models import _series
from .metrics import score


def rolling_backtest(values, model_factory, min_train=48, horizon=6, step=6):
    if any(not isinstance(v,int) or v < 1 for v in (min_train,horizon,step)):
        raise ValueError("min_train, horizon and step must be positive integers")
    data = _series(values,min_train+horizon)
    records = []
    folds = 0
    for origin in range(min_train,len(data)-horizon+1,step):
        predicted = list(model_factory().fit(data[:origin]).predict(horizon))
        if len(predicted) != horizon:
            raise ValueError("model returned the wrong forecast length")
        for lead, value in enumerate(predicted,1):
            records.append({"origin":origin, "lead":lead, "actual":data[origin+lead-1], "predicted":value})
        folds += 1
    def summarize(rows):
        return score([r["actual"] for r in rows],[r["predicted"] for r in rows])
    return {"folds":folds,"metrics":summarize(records),
            "by_horizon":{lead:summarize([r for r in records if r["lead"]==lead]) for lead in range(1,horizon+1)},
            "records":records}
