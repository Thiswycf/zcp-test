from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import uuid
from contextlib import AbstractContextManager
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from zcp_test.artifacts.jsonl import JsonlWriter
from zcp_test.config import dump_config

SCHEMA_VERSION = "1.0"
PROJECT_TIMEZONE_NAME = "Asia/Shanghai"
PROJECT_TIMEZONE = ZoneInfo(PROJECT_TIMEZONE_NAME)
RUNTIME_PACKAGES = (
    "zcp-test",
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "pandas",
    "xgboost",
    "nasbench301",
    "nats-bench",
    "timm",
)


def project_now() -> datetime:
    return datetime.now(PROJECT_TIMEZONE)


def project_now_iso() -> str:
    return project_now().isoformat()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions(packages: tuple[str, ...] = RUNTIME_PACKAGES) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions


class RunContext(AbstractContextManager["RunContext"]):
    def __init__(
        self,
        root: str | Path,
        command: list[str],
        config: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> None:
        timestamp = project_now().strftime("%Y%m%dT%H%M%S%z")
        self.run_id = uuid.uuid4().hex[:12]
        self.directory = Path(root) / f"{timestamp}_{self.run_id}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.events = JsonlWriter(self.directory / "events.jsonl", fsync_every=1)
        self.manifest_path = self.directory / "manifest.json"
        self.manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": "running",
            "command": command,
            "started_at": project_now_iso(),
            "ended_at": None,
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "pid": os.getpid(),
            "git_commit": _git_commit(Path(__file__).resolve().parents[3]),
            "environment": {
                "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "TZ": os.environ.get("TZ"),
            },
            "timezone": PROJECT_TIMEZONE_NAME,
            "runtime": runtime or {},
            "package_versions": _package_versions(),
        }
        try:
            import torch

            self.manifest["torch"] = torch.__version__
            self.manifest["cuda"] = torch.version.cuda
            self.manifest["gpus"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        except ImportError:
            self.manifest["torch"] = None
        dump_config(config, self.directory / "config.yaml")
        self._write_manifest()
        self._configure_logging()

    def ensure_directory(self, name: str) -> Path:
        if name not in {"checkpoints", "parts", "reports"}:
            raise ValueError(f"Unsupported run subdirectory: {name}")
        target = self.directory / name
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _configure_logging(self) -> None:
        self.logger = logging.getLogger(f"zcp_test.run.{self.run_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        for handler in (logging.StreamHandler(), logging.FileHandler(self.directory / "run.log")):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _write_manifest(self) -> None:
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.manifest_path)

    def event(self, kind: str, **fields: Any) -> None:
        event = {"run_id": self.run_id, "timestamp": project_now_iso(), "kind": kind, **fields}
        self.events.append(event)
        self.logger.info(
            "%s %s",
            kind,
            json.dumps(fields, ensure_ascii=False, sort_keys=True, default=str),
        )

    def close(self, status: str, error: str | None = None) -> None:
        self.manifest.update(status=status, ended_at=project_now_iso(), error=error)
        self._write_manifest()
        self.event("run_finished", status=status, error=error)
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)

    def __enter__(self) -> "RunContext":
        self.event("run_started")
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        if isinstance(exc, (InterruptedError, KeyboardInterrupt)):
            status = "interrupted"
        else:
            status = "failed" if exc else "completed"
        self.close(status, None if exc is None else str(exc))
        return False
