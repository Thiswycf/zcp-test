from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path
from typing import Any


def diagnostics(data_catalog: str | Path | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {"python": sys.version.split()[0], "python_supported": sys.version_info[:2] == (3, 10), "platform": platform.platform(), "packages": {}}
    for package in ("torch", "torchvision", "numpy", "scipy", "nats_bench", "nasbench301", "h5py", "timm"):
        report["packages"][package] = importlib.util.find_spec(package) is not None
    try:
        import torch

        report["torch"] = {"version": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda, "gpu_count": torch.cuda.device_count(), "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]}
    except ImportError:
        report["torch"] = None
    if data_catalog:
        path = Path(data_catalog).expanduser()
        report["data_catalog"] = {"path": str(path), "exists": path.exists()}
    return report

