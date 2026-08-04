#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${ZCP_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
DATA_ROOT=${ZCP_IMAGENET_ROOT:?Set ZCP_IMAGENET_ROOT to a verified ImageNet-1k root}
CANDIDATE_ROOT=${ZCP_TRAINING_CANDIDATES:?Set ZCP_TRAINING_CANDIDATES to the three architecture JSON files}
GPU_UUIDS=${ZCP_GPU_UUIDS:?Set ZCP_GPU_UUIDS to four comma-separated GPU UUIDs}
CONFIG=${ZCP_TRAINING_CONFIG:?Set ZCP_TRAINING_CONFIG to a repository-relative training config}
SPACE=${ZCP_ACCEPTANCE_SPACE:?Set ZCP_ACCEPTANCE_SPACE to the expected search-space ID}
FORMAL_EPOCHS=${ZCP_FORMAL_EPOCHS:?Set ZCP_FORMAL_EPOCHS to the complete schedule length}
FULL_DATA_EPOCHS=${ZCP_FULL_DATA_EPOCHS:?Set ZCP_FULL_DATA_EPOCHS to ceil(1% of the schedule)}
OUTPUT_ROOT=${ZCP_ACCEPTANCE_ROOT:?Set ZCP_ACCEPTANCE_ROOT inside the project runs directory}
START_AT=${ZCP_START_AT:-1}
PYTHON=${ZCP_PYTHON:-python}
TORCHRUN=${ZCP_TORCHRUN:-torchrun}
WORKERS=${ZCP_DATA_WORKERS:-8}
VALID_WORKERS=${ZCP_VALID_DATA_WORKERS:-2}
EXECUTION_STRATEGY=${ZCP_EXECUTION_STRATEGY:-sequential_ddp}
CPU_AFFINITIES=${ZCP_CPU_AFFINITIES:-}
LOCK_DIR=${XDG_CACHE_HOME:-$HOME/.cache}/zcp-test/gpu-locks
STATUS=$OUTPUT_ROOT/status.json

source "$PROJECT_ROOT/tools/acceptance/lib/launcher-runtime.sh"
acceptance_exec_immutable "$PROJECT_ROOT" "$OUTPUT_ROOT" "${BASH_SOURCE[0]}" "$@"

[[ "$START_AT" =~ ^[1-6]$ ]] || { echo "ZCP_START_AT must be in 1..6" >&2; exit 2; }
[[ "$EXECUTION_STRATEGY" =~ ^(sequential_ddp|parallel_single_gpu|packed_single_gpu)$ ]] || {
  echo "ZCP_EXECUTION_STRATEGY must be sequential_ddp, parallel_single_gpu, or packed_single_gpu" >&2
  exit 2
}
if [[ "$EXECUTION_STRATEGY" == parallel_single_gpu && "${ZCP_PARALLEL_SINGLE_GPU_ACCEPTED:-}" != yes ]]; then
  echo "parallel_single_gpu requires ZCP_PARALLEL_SINGLE_GPU_ACCEPTED=yes after a memory smoke" >&2
  exit 2
fi
if [[ "$EXECUTION_STRATEGY" == packed_single_gpu && "${ZCP_PACKED_SINGLE_GPU_ACCEPTED:-}" != yes ]]; then
  echo "packed_single_gpu requires ZCP_PACKED_SINGLE_GPU_ACCEPTED=yes after a two-process memory smoke" >&2
  exit 2
fi
[[ "$FORMAL_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "ZCP_FORMAL_EPOCHS must be positive" >&2; exit 2; }
[[ "$FULL_DATA_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "ZCP_FULL_DATA_EPOCHS must be positive" >&2; exit 2; }
(( FULL_DATA_EPOCHS * 100 >= FORMAL_EPOCHS )) || {
  echo "ZCP_FULL_DATA_EPOCHS must cover at least 1% of the formal schedule" >&2
  exit 2
}
CONFIG_PATH=$PROJECT_ROOT/$CONFIG
[[ -f "$CONFIG_PATH" ]] || { echo "Missing training config: $CONFIG_PATH" >&2; exit 2; }
for name in zcp_selected.json fixed_random.json params_flops_matched.json; do
  [[ -f "$CANDIDATE_ROOT/$name" ]] || { echo "Missing candidate: $CANDIDATE_ROOT/$name" >&2; exit 2; }
done
[[ -f "$CANDIDATE_ROOT/candidates-manifest.json" ]] || {
  echo "Missing candidate manifest: $CANDIDATE_ROOT/candidates-manifest.json" >&2
  exit 2
}
[[ -d "$DATA_ROOT/train" && -d "$DATA_ROOT/val" ]] || {
  echo "ImageNet root must contain train/ and val/: $DATA_ROOT" >&2
  exit 2
}

"$PYTHON" - "$CONFIG_PATH" "$SPACE" "$FORMAL_EPOCHS" <<'PY'
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
if config.get("space") != sys.argv[2]:
    raise SystemExit(f"config space mismatch: {config.get('space')!r} != {sys.argv[2]!r}")
if int(config.get("epochs", -1)) != int(sys.argv[3]):
    raise SystemExit(f"config epoch mismatch: {config.get('epochs')!r} != {sys.argv[3]!r}")
PY
"$PYTHON" - "$CANDIDATE_ROOT" "$SPACE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
space = sys.argv[2]
manifest = json.loads((root / "candidates-manifest.json").read_text(encoding="utf-8"))
if manifest.get("search_space_id") != space:
    raise SystemExit(
        f"candidate manifest space mismatch: {manifest.get('search_space_id')!r} != {space!r}"
    )
expected = {
    "zcp_selected.json": "zcp_selected",
    "fixed_random.json": "fixed_random",
    "params_flops_matched.json": "params_flops_matched",
}
for name, role in expected.items():
    path = root / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = manifest.get("candidates", {}).get(name, {})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if payload.get("search_space_id") != space or payload.get("candidate_role") != role:
        raise SystemExit(f"candidate identity mismatch: {name}")
    if entry.get("sha256") != digest or entry.get("role") != role:
        raise SystemExit(f"candidate manifest checksum/role mismatch: {name}")
    if entry.get("architecture_id") != payload.get("architecture_id"):
        raise SystemExit(f"candidate manifest architecture mismatch: {name}")
PY

train_classes=$(find "$DATA_ROOT/train" -mindepth 1 -maxdepth 1 -type d | wc -l)
train_files=$(find "$DATA_ROOT/train" -type f | wc -l)
val_files=$(find "$DATA_ROOT/val" -type f | wc -l)
[[ "$train_classes" == 1000 && "$train_files" == 1281167 && "$val_files" == 50000 ]] || {
  echo "Unexpected ImageNet layout: classes=$train_classes train=$train_files val=$val_files" >&2
  exit 2
}

IFS=',' read -r -a gpu_array <<< "$GPU_UUIDS"
[[ ${#gpu_array[@]} == 4 ]] || { echo "Exactly four GPU UUIDs are required" >&2; exit 2; }
cpu_affinities=()
if [[ -n "$CPU_AFFINITIES" ]]; then
  IFS=';' read -r -a cpu_affinities <<< "$CPU_AFFINITIES"
  [[ ${#cpu_affinities[@]} == 4 ]] || {
    echo "ZCP_CPU_AFFINITIES must contain four semicolon-separated CPU lists" >&2
    exit 2
  }
fi
mkdir -p "$LOCK_DIR" "$OUTPUT_ROOT/candidates"
for uuid in "${gpu_array[@]}"; do
  [[ "$uuid" =~ ^GPU-[A-Fa-f0-9-]+$ ]] || { echo "Invalid GPU UUID: $uuid" >&2; exit 2; }
done

with_gpu_lock() {
  local uuid=$1
  shift
  local descriptor
  exec {descriptor}>"$LOCK_DIR/$uuid.lock"
  flock -n "$descriptor" || {
    exec {descriptor}>&-
    echo "GPU lock unavailable: $uuid" >&2
    return 4
  }
  (
    exec {descriptor}>&-
    "$@"
  )
  local exit_code=$?
  exec {descriptor}>&-
  return "$exit_code"
}

with_all_gpu_locks() {
  local -a descriptors=()
  local uuid descriptor held
  for uuid in "${gpu_array[@]}"; do
    exec {descriptor}>"$LOCK_DIR/$uuid.lock"
    if ! flock -n "$descriptor"; then
      exec {descriptor}>&-
      for held in "${descriptors[@]}"; do
        descriptor=$held
        exec {descriptor}>&-
      done
      echo "GPU lock unavailable: $uuid" >&2
      return 4
    fi
    descriptors+=("$descriptor")
  done
  (
    for held in "${descriptors[@]}"; do
      descriptor=$held
      exec {descriptor}>&-
    done
    "$@"
  )
  local exit_code=$?
  for held in "${descriptors[@]}"; do
    descriptor=$held
    exec {descriptor}>&-
  done
  return "$exit_code"
}

for name in zcp_selected.json fixed_random.json params_flops_matched.json candidates-manifest.json; do
  source_path=$(realpath "$CANDIDATE_ROOT/$name")
  destination_path=$(realpath -m "$OUTPUT_ROOT/candidates/$name")
  if [[ "$source_path" != "$destination_path" ]]; then
    cp "$source_path" "$destination_path"
  fi
done
CANDIDATE_ROOT=$OUTPUT_ROOT/candidates

commit=${ZCP_LAUNCHER_COMMIT:-$(git -C "$PROJECT_ROOT" rev-parse HEAD)}

write_status() {
  local state=$1 current=$2 detail=$3
  "$PYTHON" - "$STATUS" "$state" "$current" "$detail" "$commit" "$DATA_ROOT" "$GPU_UUIDS" "$SPACE" "$EXECUTION_STRATEGY" <<'PY'
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

path = Path(sys.argv[1])
existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
payload = {
    **existing,
    "status": sys.argv[2],
    "current": sys.argv[3],
    "detail": sys.argv[4],
    "project_commit": sys.argv[5],
    "data_root": sys.argv[6],
    "gpu_uuids": sys.argv[7].split(","),
    "search_space_id": sys.argv[8],
    "execution_strategy": sys.argv[9],
    "pid": os.getppid(),
    "started_at": existing.get("started_at", now),
    "updated_at": now,
}
if sys.argv[2] in {"completed", "failed", "interrupted"}:
    payload["ended_at"] = now
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
os.replace(temporary, path)
PY
}

current_task=initializing
child_pids=()
stop_children() {
  for pid in "${child_pids[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
}
on_error() {
  local exit_code=$?
  stop_children
  write_status failed "$current_task" "runner failed at line $1 with exit code $exit_code"
  exit "$exit_code"
}
on_signal() {
  stop_children
  write_status interrupted "$current_task" "runner received signal"
  exit 130
}
trap 'on_error $LINENO' ERR
trap on_signal INT TERM

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU_UUIDS
export TZ=Asia/Shanghai
export OMP_NUM_THREADS=1
export PYTHONPATH=$PROJECT_ROOT/src

{
  date -Is
  findmnt -T "$DATA_ROOT"
  df -hT "$DATA_ROOT"
  printf 'space=%s ImageNet classes=%s train_files=%s val_files=%s workers=%s strategy=%s\n' \
    "$SPACE" "$train_classes" "$train_files" "$val_files" "$WORKERS" "$EXECUTION_STRATEGY"
  nvidia-smi --query-gpu=index,pci.bus_id,uuid,name,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "$OUTPUT_ROOT/preflight.log"

run_one() {
  local task_index=$1 role=$2 architecture=$3 protocol=$4 epochs=$5 fraction=$6
  if (( task_index < START_AT )); then
    printf '[%s] skipping task=%s role=%s via ZCP_START_AT=%s\n' \
      "$(date -Is)" "$task_index" "$role" "$START_AT" | tee -a "$OUTPUT_ROOT/supervisor.log"
    return
  fi
  local output=$OUTPUT_ROOT/$protocol-$role
  local launcher_log=$OUTPUT_ROOT/$protocol-$role.launcher.log
  current_task=$protocol/$role
  write_status running "$current_task" "epochs=$epochs data_fraction=$fraction task=$task_index/6"
  printf '\n[%s] starting task=%s role=%s protocol=%s epochs=%s fraction=%s\n' \
    "$(date -Is)" "$task_index" "$role" "$protocol" "$epochs" "$fraction" \
    | tee -a "$launcher_log" "$OUTPUT_ROOT/supervisor.log"
  "$TORCHRUN" --standalone --nproc-per-node=4 -m zcp_test.cli train \
    --config "$CONFIG_PATH" --acceptance-smoke --epochs "$epochs" \
    --data-fraction "$fraction" --architecture "$architecture" \
    --data-root "$DATA_ROOT" --workers "$WORKERS" --valid-workers "$VALID_WORKERS" \
    --seed 20260731 --output "$output" \
    2>&1 | tee -a "$launcher_log"
  local run
  run=$(find "$output" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
    | sort -nr | head -1 | cut -d' ' -f2-)
  "$PYTHON" - "$run/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
manifest = json.loads(path.read_text(encoding="utf-8"))
if manifest.get("status") != "completed":
    raise SystemExit(f"manifest not completed: {path}: {manifest.get('status')}")
print(f"validated {path.parent.name}: completed")
PY
}

run_one_single() {
  local task_index=$1 gpu_index=$2 role=$3 architecture=$4 protocol=$5 epochs=$6 fraction=$7
  local uuid=${gpu_array[$gpu_index]}
  if (( task_index < START_AT )); then
    printf '[%s] skipping task=%s role=%s via ZCP_START_AT=%s\n' \
      "$(date -Is)" "$task_index" "$role" "$START_AT" | tee -a "$OUTPUT_ROOT/supervisor.log"
    return
  fi
  local output=$OUTPUT_ROOT/$protocol-$role
  local launcher_log=$OUTPUT_ROOT/task-$task_index-$protocol-$role.launcher.log
  printf '\n[%s] starting task=%s gpu=%s role=%s protocol=%s epochs=%s fraction=%s\n' \
    "$(date -Is)" "$task_index" "$uuid" "$role" "$protocol" "$epochs" "$fraction" \
    | tee -a "$launcher_log" "$OUTPUT_ROOT/supervisor.log"
  local -a launch=(env CUDA_VISIBLE_DEVICES="$uuid")
  if ((${#cpu_affinities[@]})); then
    launch=(taskset -c "${cpu_affinities[$gpu_index]}" "${launch[@]}")
  fi
  "${launch[@]}" "$PYTHON" -m zcp_test.cli train \
    --config "$CONFIG_PATH" --acceptance-smoke --epochs "$epochs" \
    --data-fraction "$fraction" --architecture "$architecture" \
    --data-root "$DATA_ROOT" --workers "$WORKERS" --valid-workers "$VALID_WORKERS" \
    --seed 20260731 \
    --device cuda:0 --output "$output" 2>&1 | tee -a "$launcher_log"
  local run
  run=$(find "$output" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
    | sort -nr | head -1 | cut -d' ' -f2-)
  "$PYTHON" - "$run/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
manifest = json.loads(path.read_text(encoding="utf-8"))
if manifest.get("status") != "completed":
    raise SystemExit(f"manifest not completed: {path}: {manifest.get('status')}")
print(f"validated {path.parent.name}: completed")
PY
}

full_protocol=full-data-${FULL_DATA_EPOCHS}epoch
schedule_protocol=one-percent-data-${FORMAL_EPOCHS}epoch
write_status running initializing "validated data/config/candidates; GPU locks are acquired only while each task lane is active"
if [[ "$EXECUTION_STRATEGY" == sequential_ddp ]]; then
  with_all_gpu_locks run_one 1 zcp-selected "$CANDIDATE_ROOT/zcp_selected.json" "$full_protocol" "$FULL_DATA_EPOCHS" 1.0
  with_all_gpu_locks run_one 2 fixed-random "$CANDIDATE_ROOT/fixed_random.json" "$full_protocol" "$FULL_DATA_EPOCHS" 1.0
  with_all_gpu_locks run_one 3 params-flops-matched "$CANDIDATE_ROOT/params_flops_matched.json" "$full_protocol" "$FULL_DATA_EPOCHS" 1.0
  with_all_gpu_locks run_one 4 zcp-selected "$CANDIDATE_ROOT/zcp_selected.json" "$schedule_protocol" "$FORMAL_EPOCHS" 0.01
  with_all_gpu_locks run_one 5 fixed-random "$CANDIDATE_ROOT/fixed_random.json" "$schedule_protocol" "$FORMAL_EPOCHS" 0.01
  with_all_gpu_locks run_one 6 params-flops-matched "$CANDIDATE_ROOT/params_flops_matched.json" "$schedule_protocol" "$FORMAL_EPOCHS" 0.01
elif [[ "$EXECUTION_STRATEGY" == parallel_single_gpu ]]; then
  lane_zero() {
    run_one_single 1 0 zcp-selected "$CANDIDATE_ROOT/zcp_selected.json" "$full_protocol" "$FULL_DATA_EPOCHS" 1.0
    run_one_single 5 0 fixed-random "$CANDIDATE_ROOT/fixed_random.json" "$schedule_protocol" "$FORMAL_EPOCHS" 0.01
  }
  lane_one() {
    run_one_single 2 1 fixed-random "$CANDIDATE_ROOT/fixed_random.json" "$full_protocol" "$FULL_DATA_EPOCHS" 1.0
    run_one_single 6 1 params-flops-matched "$CANDIDATE_ROOT/params_flops_matched.json" "$schedule_protocol" "$FORMAL_EPOCHS" 0.01
  }
  write_status running parallel_tasks "four independent one-GPU lanes; each run retains its configured batch/LR protocol"
  with_gpu_lock "${gpu_array[0]}" lane_zero & child_pids+=("$!")
  with_gpu_lock "${gpu_array[1]}" lane_one & child_pids+=("$!")
  with_gpu_lock "${gpu_array[2]}" run_one_single 3 2 params-flops-matched "$CANDIDATE_ROOT/params_flops_matched.json" "$full_protocol" "$FULL_DATA_EPOCHS" 1.0 & child_pids+=("$!")
  with_gpu_lock "${gpu_array[3]}" run_one_single 4 3 zcp-selected "$CANDIDATE_ROOT/zcp_selected.json" "$schedule_protocol" "$FORMAL_EPOCHS" 0.01 & child_pids+=("$!")
  for _ in "${child_pids[@]}"; do
    wait -n
  done
else
  packed_zero() {
    local pids=()
    run_one_single 1 0 zcp-selected "$CANDIDATE_ROOT/zcp_selected.json" "$full_protocol" "$FULL_DATA_EPOCHS" 1.0 & pids+=("$!")
    run_one_single 4 0 zcp-selected "$CANDIDATE_ROOT/zcp_selected.json" "$schedule_protocol" "$FORMAL_EPOCHS" 0.01 & pids+=("$!")
    wait "${pids[@]}"
  }
  packed_one() {
    local pids=()
    run_one_single 2 1 fixed-random "$CANDIDATE_ROOT/fixed_random.json" "$full_protocol" "$FULL_DATA_EPOCHS" 1.0 & pids+=("$!")
    run_one_single 5 1 fixed-random "$CANDIDATE_ROOT/fixed_random.json" "$schedule_protocol" "$FORMAL_EPOCHS" 0.01 & pids+=("$!")
    wait "${pids[@]}"
  }
  write_status running packed_tasks "six independent runs packed onto four GPUs; batch/LR unchanged"
  with_gpu_lock "${gpu_array[0]}" packed_zero & child_pids+=("$!")
  with_gpu_lock "${gpu_array[1]}" packed_one & child_pids+=("$!")
  with_gpu_lock "${gpu_array[2]}" run_one_single 3 2 params-flops-matched "$CANDIDATE_ROOT/params_flops_matched.json" "$full_protocol" "$FULL_DATA_EPOCHS" 1.0 & child_pids+=("$!")
  with_gpu_lock "${gpu_array[3]}" run_one_single 6 3 params-flops-matched "$CANDIDATE_ROOT/params_flops_matched.json" "$schedule_protocol" "$FORMAL_EPOCHS" 0.01 & child_pids+=("$!")
  for _ in "${child_pids[@]}"; do
    wait -n
  done
fi
current_task=all
write_status completed all "all six $SPACE ImageNet acceptance runs completed"
