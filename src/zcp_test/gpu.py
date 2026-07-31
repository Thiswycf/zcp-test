from __future__ import annotations

import csv
import fcntl
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Sequence


NVIDIA_SMI_FIELDS = (
    "index",
    "uuid",
    "pci.bus_id",
    "name",
    "memory.total",
    "memory.free",
    "utilization.gpu",
)
_ACTIVE_GPU_LOCK_FDS: set[int] = set()


def _close_inherited_gpu_lock_fds() -> None:
    for descriptor in tuple(_ACTIVE_GPU_LOCK_FDS):
        try:
            os.close(descriptor)
        except OSError:
            pass
    _ACTIVE_GPU_LOCK_FDS.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_close_inherited_gpu_lock_fds)


class GPUError(RuntimeError):
    """Base error raised by GPU discovery and selection."""


class NoGPUError(GPUError):
    """Raised when no GPU satisfies the requested selection."""


class GPULockError(GPUError):
    """Raised when a GPU lock cannot be acquired."""


def _default_runner(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, **kwargs)


def _runner_stdout(result: Any) -> str:
    output = result if isinstance(result, (str, bytes)) else getattr(result, "stdout", None)
    if output is None:
        raise GPUError("nvidia-smi runner did not return stdout")
    return output.decode() if isinstance(output, bytes) else str(output)


def _integer(value: str, field: str) -> int:
    try:
        return int(value.strip())
    except ValueError as error:
        raise GPUError(f"Invalid {field} value from nvidia-smi: {value!r}") from error


def _optional_integer(value: str) -> int | None:
    value = value.strip()
    if not value or value.casefold() in {"n/a", "[n/a]", "not supported"}:
        return None
    try:
        return int(value)
    except ValueError as error:
        raise GPUError(f"Invalid numeric value from nvidia-smi: {value!r}") from error


def _pci_key(bus_id: str) -> tuple[int, int, int, int]:
    normalized = bus_id.strip().lower()
    try:
        domain_bus, device_function = normalized.rsplit(":", 1)
        if ":" in domain_bus:
            domain, bus = domain_bus.split(":", 1)
        else:
            domain, bus = "0", domain_bus
        device, function = device_function.split(".", 1)
        return (int(domain, 16), int(bus, 16), int(device, 16), int(function, 16))
    except (ValueError, TypeError) as error:
        raise GPUError(f"Invalid PCI bus ID from nvidia-smi: {bus_id!r}") from error


def enumerate_gpus(
    runner: Callable[..., Any] = _default_runner,
) -> list[dict[str, Any]]:
    """Query nvidia-smi and return GPUs sorted by PCI bus address."""

    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(NVIDIA_SMI_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = runner(command, capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError, OSError) as error:
        raise NoGPUError(f"Unable to enumerate NVIDIA GPUs: {error}") from error

    rows = csv.reader(_runner_stdout(result).splitlines(), skipinitialspace=True)
    gpus: list[dict[str, Any]] = []
    for row in rows:
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != len(NVIDIA_SMI_FIELDS):
            raise GPUError(
                f"Expected {len(NVIDIA_SMI_FIELDS)} nvidia-smi columns, got {len(row)}"
            )
        index, uuid, bus_id, model, total, free, utilization = (value.strip() for value in row)
        gpus.append(
            {
                "index": _integer(index, "GPU index"),
                "uuid": uuid,
                "bus_id": bus_id,
                "model": model,
                "memory_total_mb": _integer(total, "total memory"),
                "memory_free_mb": _integer(free, "free memory"),
                "utilization_pct": _optional_integer(utilization),
            }
        )
    return sorted(gpus, key=lambda gpu: _pci_key(gpu["bus_id"]))


def _bus_matches(actual: str, requested: str) -> bool:
    actual_key = _pci_key(actual)
    requested = requested.strip()
    if requested.count(":") == 1:
        requested = f"0000:{requested}"
    return actual_key == _pci_key(requested)


def select_gpu(
    selector: str | int | None = "auto",
    *,
    index: int | None = None,
    uuid: str | None = None,
    bus_id: str | None = None,
    model: str | None = None,
    min_free_mb: int = 0,
    runner: Callable[..., Any] = _default_runner,
    gpus: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Select one GPU and return a JSON-serializable manifest record.

    A selector may be ``auto``, an index, UUID, PCI bus ID, or model substring.
    Explicit keyword filters are also supported. Automatic selection ranks by
    free memory descending, utilization ascending, then PCI bus address.
    """

    if min_free_mb < 0:
        raise ValueError("min_free_mb must be non-negative")
    explicit = sum(value is not None for value in (index, uuid, bus_id))
    if explicit > 1:
        raise ValueError("Specify at most one of index, uuid, and bus_id")

    requested = "auto" if selector is None else selector
    if requested != "auto" and explicit:
        raise ValueError("selector cannot be combined with index, uuid, or bus_id")
    if requested != "auto":
        if isinstance(requested, int) or str(requested).strip().isdigit():
            index = int(requested)
        else:
            text = str(requested).strip()
            if text.casefold().startswith("gpu-"):
                uuid = text
            elif ":" in text and "." in text:
                bus_id = text
            else:
                if model is not None:
                    raise ValueError("model selector cannot be combined with model filter")
                model = text

    candidates = [dict(gpu) for gpu in (gpus if gpus is not None else enumerate_gpus(runner))]
    candidates = [gpu for gpu in candidates if int(gpu["memory_free_mb"]) >= min_free_mb]
    if index is not None:
        candidates = [gpu for gpu in candidates if int(gpu["index"]) == index]
    if uuid is not None:
        candidates = [gpu for gpu in candidates if str(gpu["uuid"]).casefold() == uuid.casefold()]
    if bus_id is not None:
        candidates = [gpu for gpu in candidates if _bus_matches(str(gpu["bus_id"]), bus_id)]
    if model is not None:
        model_key = model.casefold()
        candidates = [gpu for gpu in candidates if model_key in str(gpu["model"]).casefold()]
    if not candidates:
        raise NoGPUError("No NVIDIA GPU satisfies the requested selection")

    selected = min(
        candidates,
        key=lambda gpu: (
            -int(gpu["memory_free_mb"]),
            int(gpu["utilization_pct"]) if gpu.get("utilization_pct") is not None else 101,
            _pci_key(str(gpu["bus_id"])),
        ),
    )
    selection = {
        "index": int(selected["index"]),
        "uuid": str(selected["uuid"]),
        "bus_id": str(selected["bus_id"]),
        "model": str(selected["model"]),
        "memory_total_mb": int(selected["memory_total_mb"]),
        "memory_free_mb": int(selected["memory_free_mb"]),
        "utilization_pct": selected.get("utilization_pct"),
        "cuda_device_order": "PCI_BUS_ID",
        "cuda_visible_devices": str(selected["uuid"]),
    }
    return selection


def configure_cuda(
    selection: Mapping[str, Any],
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, Any]:
    """Set CUDA environment variables and return a manifest-safe selection."""

    target = os.environ if environ is None else environ
    target["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    target["CUDA_VISIBLE_DEVICES"] = str(selection["uuid"])
    configured = dict(selection)
    configured["cuda_device_order"] = target["CUDA_DEVICE_ORDER"]
    configured["cuda_visible_devices"] = target["CUDA_VISIBLE_DEVICES"]
    return configured


def gpu_lock_path(
    selection: Mapping[str, Any], cache_dir: str | Path | None = None
) -> Path:
    """Return the per-user lock path for a selected GPU."""

    if cache_dir is None:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        directory = cache_root / "zcp-test" / "gpu-locks"
    else:
        directory = Path(cache_dir).expanduser()
    identity = str(selection.get("uuid") or selection.get("bus_id") or selection["index"])
    safe_identity = "".join(
        character if character.isalnum() or character in "-." else "_"
        for character in identity
    )
    return directory / f"{safe_identity}.lock"


@contextmanager
def gpu_lock(
    selection: Mapping[str, Any],
    *,
    cache_dir: str | Path | None = None,
    timeout: float | None = None,
    poll_interval: float = 0.1,
) -> Iterator[Path]:
    """Hold an exclusive per-user file lock for a selected GPU."""

    if timeout is not None and timeout < 0:
        raise ValueError("timeout must be non-negative or None")
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")
    path = gpu_lock_path(selection, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    deadline = None if timeout is None else time.monotonic() + timeout
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                _ACTIVE_GPU_LOCK_FDS.add(handle.fileno())
                break
            except BlockingIOError as error:
                if deadline is not None and time.monotonic() >= deadline:
                    raise GPULockError(f"Timed out acquiring GPU lock: {path}") from error
                time.sleep(poll_interval)
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield path
    finally:
        try:
            if acquired:
                _ACTIVE_GPU_LOCK_FDS.discard(handle.fileno())
                handle.seek(0)
                handle.truncate()
                handle.flush()
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
