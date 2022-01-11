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


def save_csv_cache(path, observations, conflict="error"):
    """Atomically merge local CSV history. One writer per cache is required."""
    import os
    import tempfile
    from pathlib import Path
    from .csvio import read_csv, write_csv
    path = Path(path)
    existing = []
    if path.exists():
        with path.open(encoding="utf-8",newline="") as stream:
            existing = read_csv(stream)
    merged = merge_observations(existing,observations,conflict=conflict)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",newline="",dir=path.parent,suffix=".tmp",delete=False) as stream:
            temporary = stream.name
            write_csv(merged,stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return len(merged)
