from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

RUNTIME = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "acceptance"
    / "lib"
    / "launcher-runtime.sh"
)


def _metadata(lock_path: Path) -> Path:
    return Path(f"{lock_path}.lease")


def _fields(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        result[key] = value
    return result


def _owner(lock_path: Path) -> dict[str, str]:
    return _fields(lock_path)


def _wait_for(path: Path, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def _environment() -> dict[str, str]:
    return {
        **os.environ,
        "ZCP_GPU_LOCK_HEARTBEAT_SECONDS": "0.05",
        "ZCP_GPU_LOCK_LEASE_SECONDS": "1",
    }


def test_task_scope_releases_lock_while_supervisor_remains_alive(tmp_path: Path) -> None:
    lock_path = tmp_path / "gpu.lock"
    task_done = tmp_path / "task.done"
    supervisor_done = tmp_path / "supervisor.done"
    child_fds = tmp_path / "child-fds.txt"
    script = tmp_path / "supervisor.sh"
    script.write_text(
        f"""#!/usr/bin/env bash
set -Eeuo pipefail
source {RUNTIME!s}
lock_path=$1
task_done=$2
supervisor_done=$3
child_fds=$4
acceptance_with_gpu_lock "$lock_path" 1 lane-a bash -c '
  lock_path=$1
  child_fds=$2
  task_done=$3
  find /proc/$BASHPID/fd -maxdepth 1 -lname "$lock_path" -print > "$child_fds"
  sleep 0.25
  : > "$task_done"
' bash "$lock_path" "$child_fds" "$task_done"
sleep 2
: > "$supervisor_done"
""",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        ["bash", str(script), str(lock_path), str(task_done), str(supervisor_done), str(child_fds)],
        env=_environment(),
    )
    metadata = _metadata(lock_path)
    _wait_for(metadata)
    inode = lock_path.stat().st_ino
    first = _fields(metadata)

    assert first["authority"] == "kernel_flock"
    assert first["lease_enforcement"] == "observability_only_never_unlink_lock"
    assert first["state"] == "held"
    owner = _owner(lock_path)
    assert owner == {
        "pid": first["pid"],
        "host": first["host"],
        "uuid": first["uuid"],
        "acquired_at": first["acquired_at"],
    }
    assert owner["pid"] == first["owner_pid"]
    assert owner["acquired_at"].endswith("+08:00")
    observed = subprocess.run(
        [
            "bash",
            "-c",
            f"source {RUNTIME!s}; acceptance_gpu_lock_observe \"$1\"",
            "bash",
            str(lock_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "flock_held=true" in observed

    _wait_for(task_done)
    _wait_until(lambda: not metadata.exists())
    assert process.poll() is None
    assert not supervisor_done.exists()
    assert lock_path.exists()
    assert lock_path.stat().st_ino == inode
    assert lock_path.read_text(encoding="utf-8") == ""
    assert child_fds.read_text(encoding="utf-8") == ""
    subprocess.run(["flock", "-n", str(lock_path), "-c", "true"], check=True)

    process.terminate()
    process.wait(timeout=5)


def test_heartbeat_updates_owner_lease_without_changing_lock_inode(tmp_path: Path) -> None:
    lock_path = tmp_path / "gpu.lock"
    script = tmp_path / "holder.sh"
    script.write_text(
        f"""#!/usr/bin/env bash
set -Eeuo pipefail
source {RUNTIME!s}
acceptance_with_gpu_lock "$1" 1 heartbeat-holder sleep 0.6
""",
        encoding="utf-8",
    )
    process = subprocess.Popen(["bash", str(script), str(lock_path)], env=_environment())
    metadata = _metadata(lock_path)
    _wait_for(metadata)
    inode = lock_path.stat().st_ino
    first = _fields(metadata)
    first_mtime = metadata.stat().st_mtime_ns

    _wait_until(lambda: metadata.exists() and metadata.stat().st_mtime_ns > first_mtime)
    second = _fields(metadata)

    assert second["lease_id"] == first["lease_id"]
    assert second["owner_pid"] == first["owner_pid"]
    assert second["pid"] == first["pid"]
    assert second["host"] == first["host"]
    assert second["uuid"] == first["uuid"]
    assert int(second["lease_expires_epoch"]) >= int(second["heartbeat_epoch"])
    assert lock_path.stat().st_ino == inode
    assert process.wait(timeout=5) == 0
    assert lock_path.exists()
    assert lock_path.stat().st_ino == inode
    assert not metadata.exists()
    assert lock_path.read_text(encoding="utf-8") == ""


def test_concurrent_timeout_preserves_active_owner_then_reacquires(tmp_path: Path) -> None:
    lock_path = tmp_path / "gpu.lock"
    holder = subprocess.Popen(
        [
            "bash",
            "-c",
            f"source {RUNTIME!s}; acceptance_with_gpu_lock \"$1\" 1 holder sleep 0.5",
            "bash",
            str(lock_path),
        ],
        env=_environment(),
    )
    metadata = _metadata(lock_path)
    _wait_for(metadata)
    owner = _fields(metadata)
    inode = lock_path.stat().st_ino

    contender = subprocess.run(
        [
            "bash",
            "-c",
            f"source {RUNTIME!s}; acceptance_with_gpu_lock \"$1\" 0.05 contender true",
            "bash",
            str(lock_path),
        ],
        env=_environment(),
    )

    assert contender.returncode == 4
    assert _fields(metadata)["lease_id"] == owner["lease_id"]
    assert _owner(lock_path)["pid"] == owner["pid"]
    assert lock_path.stat().st_ino == inode
    assert holder.wait(timeout=5) == 0
    assert not metadata.exists()

    successor = subprocess.run(
        [
            "bash",
            "-c",
            f"source {RUNTIME!s}; acceptance_with_gpu_lock \"$1\" 1 successor true",
            "bash",
            str(lock_path),
        ],
        env=_environment(),
    )
    assert successor.returncode == 0
    assert lock_path.exists()
    assert lock_path.stat().st_ino == inode
    assert not metadata.exists()
    assert lock_path.read_text(encoding="utf-8") == ""


def test_term_signal_cleans_metadata_and_releases_kernel_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "gpu.lock"
    task_started = tmp_path / "task.started"
    task_stopped = tmp_path / "task.stopped"
    task = tmp_path / "signal-task.sh"
    task.write_text(
        """#!/usr/bin/env bash
set +e
task_started=$1
task_stopped=$2
trap 'touch "$task_stopped"; exit 143' TERM
: > "$task_started"
while true; do sleep 0.1; done
""",
        encoding="utf-8",
    )
    script = tmp_path / "signal-holder.sh"
    script.write_text(
        f"""#!/usr/bin/env bash
set +e
source {RUNTIME!s}
acceptance_with_gpu_lock "$1" 1 signal-holder bash "$2" "$3" "$4"
exit $?
""",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [
            "bash",
            str(script),
            str(lock_path),
            str(task),
            str(task_started),
            str(task_stopped),
        ],
        env=_environment(),
    )
    metadata = _metadata(lock_path)
    _wait_for(metadata)
    _wait_for(task_started)
    owner_pid = int(_fields(metadata)["owner_pid"])
    os.kill(owner_pid, signal.SIGTERM)

    assert process.wait(timeout=5) == 143
    _wait_until(lambda: not metadata.exists())
    assert task_stopped.exists()
    assert lock_path.exists()
    assert lock_path.read_text(encoding="utf-8") == ""
    subprocess.run(["flock", "-n", str(lock_path), "-c", "true"], check=True)


def test_invalid_lease_configuration_fails_before_locking(tmp_path: Path) -> None:
    lock_path = tmp_path / "gpu.lock"
    environment = _environment()
    environment["ZCP_GPU_LOCK_LEASE_SECONDS"] = "0"
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {RUNTIME!s}; acceptance_with_gpu_lock \"$1\" 1 invalid true",
            "bash",
            str(lock_path),
        ],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "positive integer" in result.stderr
    if lock_path.exists():
        subprocess.run(["flock", "-n", str(lock_path), "-c", "true"], check=True)
    assert not _metadata(lock_path).exists()


def test_nonzero_task_exit_is_propagated_after_immediate_cleanup(tmp_path: Path) -> None:
    lock_path = tmp_path / "gpu.lock"
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {RUNTIME!s}; acceptance_with_gpu_lock \"$1\" 1 failed bash -c 'exit 7'",
            "bash",
            str(lock_path),
        ],
        env=_environment(),
    )

    assert result.returncode == 7
    assert lock_path.exists()
    assert not _metadata(lock_path).exists()
    assert lock_path.read_text(encoding="utf-8") == ""
    subprocess.run(["flock", "-n", str(lock_path), "-c", "true"], check=True)
