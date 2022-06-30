"""Monthly UTC temperature climatology, fitted only on supplied history."""
from collections import defaultdict
from statistics import mean
from datetime import datetime, timezone


class MonthlyClimatology:
    def fit(self, observations):
        rows = list(observations)
        if not rows:
            raise ValueError("training observations must not be empty")
        groups = defaultdict(list)
        for row in rows:
            groups[row.timestamp.month].append(row.temperature)
        self.months = {month:mean(values) for month,values in groups.items()}
        self.fallback = mean(row.temperature for row in rows)
        return self

    def predict(self, timestamps):
        if not hasattr(self,"months"):
            raise ValueError("climatology is not fitted")
        output = []
        for stamp in timestamps:
            if not isinstance(stamp,datetime) or stamp.tzinfo is None or stamp.utcoffset() is None:
                raise ValueError("climatology query timestamps must be timezone-aware")
            output.append(self.months.get(stamp.astimezone(timezone.utc).month,self.fallback))
        return output
