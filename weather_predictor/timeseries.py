"""Leakage-aware scalar time-series transforms."""
import math


def chronological_split(values, train_fraction=0.7, validation_fraction=0.2):
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction+validation_fraction >= 1:
        raise ValueError("fractions must be positive and leave a test partition")
    data = list(values)
    a, b = int(len(data)*train_fraction), int(len(data)*(train_fraction+validation_fraction))
    if not 0 < a < b < len(data):
        raise ValueError("not enough samples for three nonempty partitions")
    return data[:a], data[a:b], data[b:]


class StandardScaler:
    """Population scaling fitted only on the caller's training samples."""
    def fit(self, values):
        from statistics import mean
        from .units import finite
        data = [finite(value) for value in values]
        if not data:
            raise ValueError("training data must not be empty")
        self.mean = mean(data)
        self.scale = math.sqrt(mean((v-self.mean)**2 for v in data)) or 1.0
        return self

    def transform(self, values):
        from .units import finite
        if not hasattr(self,"mean"):
            raise ValueError("scaler is not fitted")
        return [(finite(v)-self.mean)/self.scale for v in values]

    def inverse_transform(self, values):
        from .units import finite
        if not hasattr(self,"mean"):
            raise ValueError("scaler is not fitted")
        return [finite(v)*self.scale+self.mean for v in values]
