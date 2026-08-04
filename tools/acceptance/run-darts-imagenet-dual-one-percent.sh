#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${ZCP_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
DATA_ROOT=${ZCP_IMAGENET_ROOT:?Set ZCP_IMAGENET_ROOT to a verified ImageNet-1k root}
CANDIDATE_ROOT=${ZCP_DARTS_CANDIDATES:?Set ZCP_DARTS_CANDIDATES to the three architecture JSON files}
GPU_UUIDS=${ZCP_GPU_UUIDS:?Set ZCP_GPU_UUIDS to four comma-separated GPU UUIDs}
OUTPUT_ROOT=${ZCP_ACCEPTANCE_ROOT:-$PROJECT_ROOT/runs/acceptance/darts-imagenet}
START_AT=${ZCP_START_AT:-1}
PYTHON=${ZCP_PYTHON:-python}
TORCHRUN=${ZCP_TORCHRUN:-torchrun}
WORKERS=${ZCP_DATA_WORKERS:-8}
VALID_WORKERS=${ZCP_VALID_DATA_WORKERS:-2}
LOCK_DIR=${XDG_CACHE_HOME:-$HOME/.cache}/zcp-test/gpu-locks
STATUS=$OUTPUT_ROOT/status.json

source "$PROJECT_ROOT/tools/acceptance/lib/launcher-runtime.sh"
acceptance_exec_immutable "$PROJECT_ROOT" "$OUTPUT_ROOT" "${BASH_SOURCE[0]}" "$@"

[[ "$START_AT" =~ ^[1-6]$ ]] || { echo "ZCP_START_AT must be in 1..6" >&2; exit 2; }
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

for name in zcp_selected.json fixed_random.json params_matched_random_pool.json; do
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
  "$PYTHON" - "$STATUS" "$state" "$current" "$detail" "$commit" "$DATA_ROOT" "$GPU_UUIDS" <<'PY'
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
on_error() {
  local exit_code=$?
  write_status failed "$current_task" "runner failed at line $1 with exit code $exit_code"
  exit "$exit_code"
}
on_signal() {
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
  printf 'ImageNet classes=%s train_files=%s val_files=%s workers_per_rank=%s\n' \
    "$train_classes" "$train_files" "$val_files" "$WORKERS"
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
    --config "$PROJECT_ROOT/configs/training/darts_imagenet.yaml" \
    --acceptance-smoke --epochs "$epochs" --data-fraction "$fraction" \
    --architecture "$architecture" --data-root "$DATA_ROOT" --workers "$WORKERS" \
    --valid-workers "$VALID_WORKERS" \
    --seed 20260731 --output "$output" 2>&1 | tee -a "$launcher_log"
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

write_status running initializing "validated fast data root; four GPU locks are held only while each DDP task is active"
with_all_gpu_locks run_one 1 zcp-selected "$CANDIDATE_ROOT/zcp_selected.json" full-data-3epoch 3 1.0
with_all_gpu_locks run_one 2 fixed-random "$CANDIDATE_ROOT/fixed_random.json" full-data-3epoch 3 1.0
with_all_gpu_locks run_one 3 params-matched "$CANDIDATE_ROOT/params_matched_random_pool.json" full-data-3epoch 3 1.0
with_all_gpu_locks run_one 4 zcp-selected "$CANDIDATE_ROOT/zcp_selected.json" one-percent-data-250epoch 250 0.01
with_all_gpu_locks run_one 5 fixed-random "$CANDIDATE_ROOT/fixed_random.json" one-percent-data-250epoch 250 0.01
with_all_gpu_locks run_one 6 params-matched "$CANDIDATE_ROOT/params_matched_random_pool.json" one-percent-data-250epoch 250 0.01
current_task=all
write_status completed all "all six DARTS ImageNet acceptance runs completed"
