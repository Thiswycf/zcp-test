from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


class JsonlWriter:
    def __init__(self, path: str | Path, fsync_every: int = 100) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync_every = max(1, fsync_every)
        self._pending = 0
        self._lock = threading.Lock()

    def append(self, record: Mapping[str, Any]) -> None:
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()
            self._pending += 1
            if self._pending >= self.fsync_every:
                os.fsync(handle.fileno())
                self._pending = 0


def read_jsonl(path: str | Path, tolerate_partial_last_line: bool = True) -> Iterator[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return
    lines = source.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if tolerate_partial_last_line and index == len(lines) - 1:
                break
            raise
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record {index + 1} is not an object")
        yield value


def merge_jsonl(parts: Iterable[str | Path], destination: str | Path, key_fields: tuple[str, ...]) -> int:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for part in parts:
        for record in read_jsonl(part):
            key = tuple(record.get(field) for field in key_fields)
            if None in key:
                raise ValueError(f"Missing merge key in {part}: {key_fields}")
            unique[key] = record
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(unique, key=lambda item: tuple(map(str, item))):
            handle.write(json.dumps(unique[key], ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)
    return len(unique)

