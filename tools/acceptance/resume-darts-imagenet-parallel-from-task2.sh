#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA_ROOT=${ZCP_IMAGENET_ROOT:?Set ZCP_IMAGENET_ROOT to a verified ImageNet-1k root}
CANDIDATE_ROOT=${ZCP_DARTS_CANDIDATES:?Set ZCP_DARTS_CANDIDATES to the three architecture JSON files}
GPU_UUIDS=${ZCP_GPU_UUIDS:?Set ZCP_GPU_UUIDS to four comma-separated GPU UUIDs}
OUTPUT_ROOT=${ZCP_ACCEPTANCE_ROOT:-$PROJECT_ROOT/runs/acceptance/darts-imagenet-parallel}
PYTHON=${ZCP_PYTHON:-python}
WORKERS=${ZCP_DATA_WORKERS_PER_TASK:-12}
VALID_WORKERS=${ZCP_VALID_DATA_WORKERS_PER_TASK:-2}
LOCK_DIR=${XDG_CACHE_HOME:-$HOME/.cache}/zcp-test/gpu-locks
STATUS=$OUTPUT_ROOT/status.json
CONFIG=$PROJECT_ROOT/configs/training/darts_imagenet.yaml

for name in zcp_selected.json fixed_random.json params_matched_random_pool.json; do
  [[ -f "$CANDIDATE_ROOT/$name" ]] || { echo "Missing candidate: $CANDIDATE_ROOT/$name" >&2; exit 2; }
done
[[ -d "$DATA_ROOT/train" && -d "$DATA_ROOT/val" ]] || {
  echo "ImageNet root must contain train/ and val/: $DATA_ROOT" >&2
  exit 2
}
train_classes=$(find "$DATA_ROOT/train" -mindepth 1 -maxdepth 1 -type d | wc -l)
train_files=$(find "$DATA_ROOT/train" -type f | wc -l)
val_files=$(find "$DATA_ROOT/val" -type f | wc -l)
[[ "$train_classes" == 1000 && "$train_files" == 1281167 && "$val_files" == 50000 ]] || {
  echo "Unexpected ImageNet layout: classes=$train_classes train=$train_files val=$val_files" >&2
  exit 2
}

IFS=',' read -r -a gpu_array <<< "$GPU_UUIDS"
[[ ${#gpu_array[@]} == 4 ]] || { echo "Exactly four GPU UUIDs are required" >&2; exit 2; }
mkdir -p "$LOCK_DIR" "$OUTPUT_ROOT/candidates"
for uuid in "${gpu_array[@]}"; do
  [[ "$uuid" =~ ^GPU-[A-Fa-f0-9-]+$ ]] || { echo "Invalid GPU UUID: $uuid" >&2; exit 2; }
done
for name in zcp_selected.json fixed_random.json params_matched_random_pool.json; do
  source_path=$(realpath "$CANDIDATE_ROOT/$name")
  destination_path=$(realpath -m "$OUTPUT_ROOT/candidates/$name")
  if [[ "$source_path" != "$destination_path" ]]; then
    cp "$source_path" "$destination_path"
  fi
done
CANDIDATE_ROOT=$OUTPUT_ROOT/candidates

commit=$(git -C "$PROJECT_ROOT" rev-parse HEAD)
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || {
  echo "Project worktree must be clean before acceptance training" >&2
  exit 2
}

write_status() {
  local state=$1 detail=$2
  "$PYTHON" - "$STATUS" "$state" "$detail" "$commit" "$DATA_ROOT" "$GPU_UUIDS" <<'PY'
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
    "current": "parallel_tasks_2_to_6",
    "detail": sys.argv[3],
    "project_commit": sys.argv[4],
    "data_root": sys.argv[5],
    "gpu_uuids": sys.argv[6].split(","),
    "execution_strategy": "longest_first_per_lane_locks_global_batch_128",
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

child_pids=()
stop_children() {
  for pid in "${child_pids[@]:-}"; do
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
  done
}
on_error() {
  local exit_code=$?
  stop_children
  write_status failed "parallel lane failed at line $1 with exit code $exit_code"
  exit "$exit_code"
}
on_signal() {
  stop_children
  write_status interrupted "parallel runner received signal"
  exit 130
}
trap 'on_error $LINENO' ERR
trap on_signal INT TERM

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TZ=Asia/Shanghai
export OMP_NUM_THREADS=1
export PYTHONPATH=$PROJECT_ROOT/src

{
  date -Is
  findmnt -T "$DATA_ROOT"
  df -hT "$DATA_ROOT"
  printf 'ImageNet classes=%s train_files=%s val_files=%s workers_per_task=%s strategy=4x-single-gpu\n' \
    "$train_classes" "$train_files" "$val_files" "$WORKERS"
  nvidia-smi --query-gpu=index,pci.bus_id,uuid,name,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} | tee "$OUTPUT_ROOT/preflight.log"

run_single() {
  local task_index=$1 uuid=$2 role=$3 architecture=$4 protocol=$5 epochs=$6 fraction=$7
  local output=$OUTPUT_ROOT/$protocol-$role
  local launcher_log=$OUTPUT_ROOT/task-${task_index}-$protocol-$role.launcher.log
  printf '[%s] task=%s gpu=%s role=%s protocol=%s epochs=%s fraction=%s\n' \
    "$(date -Is)" "$task_index" "$uuid" "$role" "$protocol" "$epochs" "$fraction" \
    | tee -a "$launcher_log" "$OUTPUT_ROOT/supervisor.log"
  CUDA_VISIBLE_DEVICES=$uuid "$PYTHON" -m zcp_test.cli train \
    --config "$CONFIG" --acceptance-smoke --epochs "$epochs" --data-fraction "$fraction" \
    --architecture "$architecture" --data-root "$DATA_ROOT" --workers "$WORKERS" \
    --valid-workers "$VALID_WORKERS" \
    --seed 20260731 --device cuda:0 --output "$output" 2>&1 | tee -a "$launcher_log"
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

with_gpu_lock() {
  local uuid=$1
  shift
  local descriptor
  exec {descriptor}>"$LOCK_DIR/$uuid.lock"
  flock -n "$descriptor" || { echo "GPU lock unavailable: $uuid" >&2; exit 4; }
  (
    exec {descriptor}>&-
    "$@"
  )
}

short_lane() {
  local uuid=$1
  run_single 2 "$uuid" fixed-random "$CANDIDATE_ROOT/fixed_random.json" full-data-3epoch 3 1.0
  run_single 3 "$uuid" params-matched "$CANDIDATE_ROOT/params_matched_random_pool.json" full-data-3epoch 3 1.0
}

write_status running "longest tasks 4-6 start first; task 2-3 share the fourth lane; each lane releases its own lock when complete"
with_gpu_lock "${gpu_array[0]}" run_single 4 "${gpu_array[0]}" zcp-selected "$CANDIDATE_ROOT/zcp_selected.json" one-percent-data-250epoch 250 0.01 & child_pids+=("$!")
with_gpu_lock "${gpu_array[1]}" run_single 5 "${gpu_array[1]}" fixed-random "$CANDIDATE_ROOT/fixed_random.json" one-percent-data-250epoch 250 0.01 & child_pids+=("$!")
with_gpu_lock "${gpu_array[2]}" run_single 6 "${gpu_array[2]}" params-matched "$CANDIDATE_ROOT/params_matched_random_pool.json" one-percent-data-250epoch 250 0.01 & child_pids+=("$!")
with_gpu_lock "${gpu_array[3]}" short_lane "${gpu_array[3]}" & child_pids+=("$!")

for _ in "${child_pids[@]}"; do
  wait -n
done
write_status completed "tasks 2-6 completed with longest-first lane scheduling; task 1 remains the prior completed run"
