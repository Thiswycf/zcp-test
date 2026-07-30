from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


class JsonlTable:
    """Strict, pickle-free JSONL reader used by runtime benchmark adapters."""

    def __init__(self, path: str | Path, *, expected_kind: str | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"JSONL benchmark does not exist: {self.path}")
        self.expected_kind = expected_kind

    def records(self) -> Iterator[dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object at {self.path}:{line_number}")
                if self.expected_kind and value.get("record_kind") != self.expected_kind:
                    raise ValueError(
                        f"Unexpected record kind at {self.path}:{line_number}: "
                        f"{value.get('record_kind')!r}"
                    )
                yield value

    def load(self) -> list[dict[str, Any]]:
        return list(self.records())


def write_jsonl_atomic(records: Iterable[Mapping[str, Any]], destination: str | Path) -> Path:
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(dict(record), handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
        handle.flush()
    temporary.replace(path)
    return path


def convert_trusted_torch_records(
    source: str | Path,
    destination: str | Path,
    converter: Callable[[Any], Iterable[Mapping[str, Any]]],
    *,
    trusted: bool,
) -> Path:
    """Convert a trusted torch/pickle release once; runtime code reads only JSONL."""
    if not trusted:
        raise PermissionError("Torch/pickle conversion requires explicit trusted=True")
    import torch

    try:
        payload = torch.load(Path(source).expanduser(), map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "Trusted torch conversion requires torch.load(weights_only=...) support; "
            "refusing unsafe pickle fallback"
        ) from error
    return write_jsonl_atomic(converter(payload), destination)
