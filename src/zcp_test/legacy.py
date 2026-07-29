from __future__ import annotations

from pathlib import Path
from typing import Any

from zcp_test.artifacts import JsonlWriter


def import_pickle(source: str | Path, destination: str | Path, trusted: bool = False) -> int:
    if not trusted:
        raise PermissionError("Legacy pickle import requires --trusted")
    import pickle

    with Path(source).open("rb") as handle:
        payload = pickle.load(handle)
    writer = JsonlWriter(destination, fsync_every=1)
    records: list[dict[str, Any]]
    if isinstance(payload, list):
        records = [value if isinstance(value, dict) else {"value": value} for value in payload]
    elif isinstance(payload, dict):
        records = [{"key": key, "value": value} for key, value in payload.items()]
    else:
        records = [{"value": payload}]
    for index, record in enumerate(records):
        writer.append({"legacy_index": index, **record})
    return len(records)

