from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def rng_state() -> dict[str, Any]:
    import torch

    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: str | Path, trusted: bool = False) -> dict[str, Any]:
    if not trusted:
        raise PermissionError("Checkpoint loading requires trusted=True")
    import torch

    value = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError("Checkpoint must contain a mapping")
    return value
