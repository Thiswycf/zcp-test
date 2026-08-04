from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from zcp_test.artifacts import JsonlWriter


def import_pickle(source: str | Path, destination: str | Path, trusted: bool = False) -> int:
    if not trusted:
        raise PermissionError("Legacy pickle import requires --trusted")
    destination_path = Path(destination)
    if destination_path.exists():
        raise FileExistsError(
            f"Legacy import output already exists; choose a new path: {destination_path}"
        )
    import pickle

    with Path(source).open("rb") as handle:
        payload = pickle.load(handle)
    records: list[dict[str, Any]]
    if isinstance(payload, list):
        records = [value if isinstance(value, dict) else {"value": value} for value in payload]
    elif isinstance(payload, dict):
        records = [{"key": key, "value": value} for key, value in payload.items()]
    else:
        records = [{"value": payload}]
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination_path.parent,
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        writer = JsonlWriter(temporary, fsync_every=1)
        for index, record in enumerate(records):
            writer.append({"legacy_index": index, **record})
        try:
            os.link(temporary, destination_path)
        except FileExistsError as error:
            raise FileExistsError(
                f"Legacy import output already exists; choose a new path: {destination_path}"
            ) from error
        directory_descriptor = os.open(destination_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return len(records)
