import json
import os
import signal
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from zcp_test import cli
from zcp_test.gpu import (
    GPULockError,
    NoGPUError,
    configure_cuda,
    enumerate_gpus,
    gpu_lock,
    select_gpu,
)


NVIDIA_SMI_OUTPUT = """\
1, GPU-4090D, 00000000:65:00.0, NVIDIA GeForce RTX 4090 D, 24564, 22000, 8
0, GPU-4090, 00000000:17:00.0, NVIDIA GeForce RTX 4090, 24564, 18000, 2
2, GPU-4090D-BUSY, 00000000:B3:00.0, NVIDIA GeForce RTX 4090 D, 24564, 22000, 41
"""


class MockRunner:
    def __init__(self, stdout=NVIDIA_SMI_OUTPUT):
        self.stdout = stdout
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return SimpleNamespace(stdout=self.stdout)


def test_enumerate_gpus_queries_csv_and_sorts_by_pci():
    runner = MockRunner()

    gpus = enumerate_gpus(runner)

    assert [gpu["index"] for gpu in gpus] == [0, 1, 2]
    assert gpus[1] == {
        "index": 1,
        "uuid": "GPU-4090D",
        "bus_id": "00000000:65:00.0",
        "model": "NVIDIA GeForce RTX 4090 D",
        "memory_total_mb": 24564,
        "memory_free_mb": 22000,
        "utilization_pct": 8,
    }
    command, kwargs = runner.calls[0]
    assert command[0] == "nvidia-smi"
    assert command[-1] == "--format=csv,noheader,nounits"
    assert "pci.bus_id" in command[1]
    assert kwargs == {"capture_output": True, "text": True, "check": True}


def test_auto_selection_ranks_free_memory_then_utilization_then_bus():
    selection = select_gpu(runner=MockRunner())

    assert selection["uuid"] == "GPU-4090D"
    assert selection["memory_free_mb"] == 22000
    assert selection["cuda_device_order"] == "PCI_BUS_ID"
    assert selection["cuda_visible_devices"] == "GPU-4090D"
    json.dumps(selection)


@pytest.mark.parametrize(
    ("kwargs", "expected_uuid"),
    [
        ({"index": 0}, "GPU-4090"),
        ({"uuid": "gpu-4090d-busy"}, "GPU-4090D-BUSY"),
        ({"bus_id": "65:00.0"}, "GPU-4090D"),
        ({"model": "4090 D"}, "GPU-4090D"),
    ],
)
def test_selection_supports_index_uuid_bus_and_model(kwargs, expected_uuid):
    assert select_gpu(runner=MockRunner(), **kwargs)["uuid"] == expected_uuid


def test_selector_shorthand_and_minimum_free_memory():
    runner = MockRunner()

    assert select_gpu(0, runner=runner)["uuid"] == "GPU-4090"
    assert select_gpu("GPU-4090D-BUSY", runner=runner)["index"] == 2
    assert select_gpu("17:00.0", runner=runner)["index"] == 0
    with pytest.raises(NoGPUError, match="satisfies"):
        select_gpu(index=0, min_free_mb=20000, runner=runner)


def test_no_gpus_fails_cleanly():
    with pytest.raises(NoGPUError, match="No NVIDIA GPU"):
        select_gpu(runner=MockRunner(""))


def test_configure_cuda_sets_environment_and_returns_copy():
    selection = select_gpu(runner=MockRunner())
    environment = {}

    configured = configure_cuda(selection, environment)

    assert environment == {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "GPU-4090D",
    }
    assert configured == selection
    assert configured is not selection


def test_gpu_lock_is_exclusive_and_released(tmp_path):
    selection = select_gpu(runner=MockRunner())

    with gpu_lock(selection, cache_dir=tmp_path, timeout=0) as path:
        assert path.parent == tmp_path
        assert path.read_text(encoding="utf-8").startswith("pid=")
        with pytest.raises(GPULockError, match="Timed out"):
            with gpu_lock(selection, cache_dir=tmp_path, timeout=0):
                pass

    assert path.read_text(encoding="utf-8") == ""
    with gpu_lock(selection, cache_dir=tmp_path, timeout=0) as reacquired:
        assert reacquired == path


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_gpu_lock_fd_is_not_retained_by_fork_child(tmp_path):
    selection = select_gpu(runner=MockRunner())
    read_descriptor, write_descriptor = os.pipe()

    with gpu_lock(selection, cache_dir=tmp_path, timeout=0):
        child_pid = os.fork()
        if child_pid == 0:
            os.close(read_descriptor)
            os.write(write_descriptor, b"ready")
            time.sleep(5)
            os._exit(0)
        os.close(write_descriptor)
        assert os.read(read_descriptor, 5) == b"ready"
        os.close(read_descriptor)

    try:
        with gpu_lock(selection, cache_dir=tmp_path, timeout=0):
            pass
    finally:
        os.kill(child_pid, signal.SIGTERM)
        os.waitpid(child_pid, 0)


def test_auto_device_selection_skips_a_locked_best_gpu(monkeypatch):
    gpus = enumerate_gpus(MockRunner())
    best = select_gpu(gpus=gpus)
    locked_uuid = best["uuid"]

    @contextmanager
    def fake_lock(selection, **_kwargs):
        if selection["uuid"] == locked_uuid:
            raise GPULockError("locked")
        yield None

    args = SimpleNamespace(
        device=None,
        gpu="auto",
        gpu_model=None,
        min_free_memory=0,
        gpu_lock_timeout=0.0,
        _gpu_selection=best,
    )
    monkeypatch.setattr(cli, "enumerate_gpus", lambda: gpus)
    monkeypatch.setattr(cli, "gpu_lock", fake_lock)
    monkeypatch.setattr(cli, "_device", lambda name: name)

    with cli._selected_device(args) as (device, selection):
        assert device == "cuda:0"
        assert selection["uuid"] == "GPU-4090D-BUSY"
        assert selection["selection_strategy"] == "auto"
        assert selection["nvidia_smi_index"] == 2
        assert selection["torch_logical_index"] == 0


def test_auto_device_selection_probes_all_candidates_before_waiting(monkeypatch):
    gpus = enumerate_gpus(MockRunner())
    best = select_gpu(gpus=gpus)
    attempts = []

    @contextmanager
    def fake_lock(selection, *, timeout):
        attempts.append((selection["uuid"], timeout))
        if selection["uuid"] == best["uuid"]:
            raise GPULockError("locked")
        yield None

    args = SimpleNamespace(
        device=None,
        gpu="auto",
        gpu_model=None,
        min_free_memory=0,
        gpu_lock_timeout=120.0,
        _gpu_selection=best,
    )
    monkeypatch.setattr(cli, "enumerate_gpus", lambda: gpus)
    monkeypatch.setattr(cli, "gpu_lock", fake_lock)
    monkeypatch.setattr(cli, "_device", lambda name: name)

    with cli._selected_device(args) as (_, selection):
        assert selection["uuid"] == "GPU-4090D-BUSY"

    assert attempts == [("GPU-4090D", 0.0), ("GPU-4090D-BUSY", 0.0)]


def test_auto_device_selection_polls_candidates_with_one_global_timeout(monkeypatch):
    gpus = enumerate_gpus(MockRunner())
    best = select_gpu(gpus=gpus)
    attempts = []
    sleeps = []

    @contextmanager
    def fake_lock(selection, *, timeout):
        attempts.append((selection["uuid"], timeout))
        if len(attempts) <= len(gpus) or selection["uuid"] != "GPU-4090D-BUSY":
            raise GPULockError("locked")
        yield None

    args = SimpleNamespace(
        device=None,
        gpu="auto",
        gpu_model=None,
        min_free_memory=0,
        gpu_lock_timeout=120.0,
        _gpu_selection=best,
    )
    monkeypatch.setattr(cli, "enumerate_gpus", lambda: gpus)
    monkeypatch.setattr(cli, "gpu_lock", fake_lock)
    monkeypatch.setattr(cli, "_device", lambda name: name)
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)

    with cli._selected_device(args) as (_, selection):
        assert selection["uuid"] == "GPU-4090D-BUSY"

    assert attempts[:3] == [
        ("GPU-4090D", 0.0),
        ("GPU-4090D-BUSY", 0.0),
        ("GPU-4090", 0.0),
    ]
    assert attempts[3] == ("GPU-4090D-BUSY", 0.0)
    assert sleeps == [0.1]


def test_explicit_gpu_selection_waits_for_requested_timeout(monkeypatch):
    gpus = enumerate_gpus(MockRunner())
    selection = select_gpu(uuid="GPU-4090", gpus=gpus)
    attempts = []

    @contextmanager
    def fake_lock(candidate, *, timeout):
        attempts.append((candidate["uuid"], timeout))
        raise GPULockError("locked")
        yield None

    args = SimpleNamespace(
        device=None,
        gpu="GPU-4090",
        gpu_model=None,
        min_free_memory=0,
        gpu_lock_timeout=120.0,
        _gpu_selection=selection,
    )
    monkeypatch.setattr(cli, "gpu_lock", fake_lock)

    with pytest.raises(GPULockError, match="selected GPU"):
        with cli._selected_device(args):
            pass

    assert attempts == [("GPU-4090", 120.0)]


def test_gpu_selection_does_not_swallow_body_lock_errors(monkeypatch):
    gpus = enumerate_gpus(MockRunner())
    args = SimpleNamespace(
        device=None,
        gpu="auto",
        gpu_model=None,
        min_free_memory=0,
        gpu_lock_timeout=0.0,
        _gpu_selection=select_gpu(gpus=gpus),
    )
    monkeypatch.setattr(cli, "enumerate_gpus", lambda: gpus)
    monkeypatch.setattr(cli, "_device", lambda name: name)

    with pytest.raises(GPULockError, match="task failure"):
        with cli._selected_device(args):
            raise GPULockError("task failure")
