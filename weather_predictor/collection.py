"""Deterministic collection merging. Conflicting duplicates never disappear silently."""


def merge_observations(*batches, conflict="error"):
    if conflict not in ("error", "last"):
        raise ValueError("conflict must be error or last")
    indexed = {}
    for batch in batches:
        for item in batch:
            previous = indexed.get(item.timestamp)
            if previous is not None and previous != item and conflict == "error":
                raise ValueError("conflicting observations at " + item.timestamp.isoformat())
            indexed[item.timestamp] = item
    return [indexed[key] for key in sorted(indexed)]
