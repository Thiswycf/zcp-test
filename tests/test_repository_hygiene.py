from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DATA_SUFFIXES = {
    ".tfrecord",
    ".pth",
    ".pt",
    ".pkl",
    ".pickle",
    ".pbz2",
    ".tar",
}
MACHINE_PATHS = (b"/home/" + b"lqz25zhj", b"/public/" + b"zhanghaojie")
SECRET_MARKERS = (
    b"gh" + b"p_",
    b"github_" + b"pat_",
    b"AK" + b"IA",
    b"BEGIN OPENSSH" + b" PRIVATE KEY",
)


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def test_repository_contains_no_large_data_or_secrets() -> None:
    violations: list[str] = []
    for path in _tracked_files():
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.stat().st_size > 10 * 1024 * 1024:
            violations.append(f"large file: {relative}")
        if path.suffix.lower() in FORBIDDEN_DATA_SUFFIXES:
            violations.append(f"benchmark/checkpoint data: {relative}")
        payload = path.read_bytes()
        if any(machine_path in payload for machine_path in MACHINE_PATHS):
            violations.append(f"machine-specific path: {relative}")
        if any(marker in payload for marker in SECRET_MARKERS):
            violations.append(f"possible credential: {relative}")
    assert not violations, "\n".join(violations)
