"""Describe missing data without automatically modifying the observations."""
from datetime import timedelta
from .csvio import COLUMNS


def quality_report(observations, step=timedelta(hours=1)):
    if step.total_seconds() <= 0:
        raise ValueError("step must be positive")
    rows = list(observations)
    stamps = sorted({item.timestamp for item in rows})
    gaps = [b-a for a,b in zip(stamps, stamps[1:])]
    return {
        "count": len(rows), "unique": len(stamps), "duplicates": len(rows)-len(stamps),
        "start": stamps[0].isoformat() if stamps else None,
        "end": stamps[-1].isoformat() if stamps else None,
        "missing_steps": sum(max(0, int(gap // step)-1) for gap in gaps),
        "irregular_intervals": sum(gap % step != timedelta(0) for gap in gaps),
        "missing": {name: sum(getattr(item,name) is None for item in rows) for name in COLUMNS[2:]},
    }


def outlier_indices(values, window=3, threshold=3.0):
    """Centered Hampel-style diagnostic, not a causal online filter."""
    import math
    from statistics import median
    from .units import finite
    threshold = finite(threshold)
    if not isinstance(window,int) or window < 1 or threshold <= 0:
        raise ValueError("window and threshold must be positive")
    data = [finite(value) for value in values]
    result = []
    for i, value in enumerate(data):
        local = data[max(0,i-window):i+window+1]
        center = median(local)
        scale = 1.4826 * median(abs(v-center) for v in local)
        if abs(value-center) > threshold*max(scale, 1e-12):
            result.append(i)
    return result
