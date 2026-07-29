from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable, Mapping

from zcp_test.types import Architecture


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def stable_architecture_id(search_space_id: str, specification: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(specification).encode("utf-8")).hexdigest()[:20]
    return f"{search_space_id}:{digest}"


def architecture_from_spec(
    search_space_id: str, specification: Mapping[str, Any], benchmark_index: int | None = None
) -> Architecture:
    return Architecture(
        search_space_id=search_space_id,
        architecture_id=stable_architecture_id(search_space_id, specification),
        spec=specification,
        benchmark_index=benchmark_index,
    )


def ensure_path(path: str | Path, *, kind: str = "path") -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Configured {kind} does not exist: {resolved}")
    return resolved


def bounded_range(length: int, start: int, end: int | None) -> range:
    if start < 0:
        raise ValueError("start must be non-negative")
    stop = length if end is None else end
    if stop < start or stop > length:
        raise ValueError(f"Invalid architecture range [{start}, {stop}) for length {length}")
    return range(start, stop)


def sample_index(length: int, seed: int | None) -> int:
    if length <= 0:
        raise ValueError("Cannot sample from an empty benchmark")
    return random.Random(seed).randrange(length)


def require_architecture_space(architecture: Architecture, search_space_id: str) -> None:
    if architecture.search_space_id != search_space_id:
        raise ValueError(
            f"Architecture belongs to {architecture.search_space_id!r}, expected {search_space_id!r}"
        )


ModelBuilder = Callable[[Architecture, str], Any]
