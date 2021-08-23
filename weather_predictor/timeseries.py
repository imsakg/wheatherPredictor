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
