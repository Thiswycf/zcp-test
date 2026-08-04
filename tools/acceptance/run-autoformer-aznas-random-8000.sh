#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${ZCP_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${ZCP_PYTHON:-python}
GPU_UUIDS=${ZCP_GPU_UUIDS:?Set ZCP_GPU_UUIDS to two comma-separated GPU UUIDs}
OUTPUT_ROOT=${ZCP_ACCEPTANCE_ROOT:-$PROJECT_ROOT/runs/acceptance/autoformer-aznas-random-8000}
LOCK_DIR=${XDG_CACHE_HOME:-$HOME/.cache}/zcp-test/gpu-locks
LOCK_TIMEOUT=${ZCP_GPU_LOCK_TIMEOUT_SECONDS:-7200}
POPULATION=${ZCP_AZNAS_POPULATION:-8000}
STATUS=$OUTPUT_ROOT/status.json
SUPERVISOR_LOG=$OUTPUT_ROOT/supervisor.log
SEEDS=(20260731 20260732 20260733)

source "$PROJECT_ROOT/tools/acceptance/lib/launcher-runtime.sh"
acceptance_exec_immutable "$PROJECT_ROOT" "$OUTPUT_ROOT" "${BASH_SOURCE[0]}" "$@"

IFS=',' read -r -a gpu_array <<< "$GPU_UUIDS"
[[ ${#gpu_array[@]} == 2 ]] || { echo "Exactly two GPU UUIDs are required" >&2; exit 2; }
for uuid in "${gpu_array[@]}"; do
  [[ "$uuid" =~ ^GPU-[A-Fa-f0-9-]+$ ]] || { echo "Invalid GPU UUID: $uuid" >&2; exit 2; }
done
[[ "$POPULATION" =~ ^[1-9][0-9]*$ ]] || { echo "Population must be positive" >&2; exit 2; }

mkdir -p "$LOCK_DIR" "$OUTPUT_ROOT"
commit=${ZCP_LAUNCHER_COMMIT:-$(git -C "$PROJECT_ROOT" rev-parse HEAD)}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TZ=Asia/Shanghai
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONPATH=$PROJECT_ROOT/src

supervisor_event() {
  local event=$1
  shift
  printf '[%s] event=%q pid=%s snapshot=%q %s\n' \
    "$(date -Is)" "$event" "$$" "${ZCP_LAUNCHER_SNAPSHOT:-unknown}" "$*" \
    >> "$SUPERVISOR_LOG"
}

write_status() {
  local state=$1 detail=$2
  "$PYTHON" - "$STATUS" "$state" "$detail" "$commit" "$GPU_UUIDS" "$POPULATION" <<'PY'
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
    "detail": sys.argv[3],
    "project_commit": sys.argv[4],
    "gpu_uuids": sys.argv[5].split(","),
    "population_per_seed": int(sys.argv[6]),
    "seeds": [20260731, 20260732, 20260733],
    "primary_selection_seed": 20260731,
    "supporting_robustness_seeds": [20260732, 20260733],
    "candidate_selection_protocol": "predeclared_primary_run_supporting_seed_robustness_v1",
    "execution_strategy": "packed_2_plus_1_on_two_gpus",
    "model_initialization_protocol": "architecture-hash-v1",
    "search_protocol": "aznas-upstream-8000-random-candidates-project-sampler-v1",
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

latest_state() {
  local seed_root=$1
  find "$seed_root" -type f -name search-state.json -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d' ' -f2-
}

completed_run_exists() {
  local seed_root=$1
  "$PYTHON" - "$seed_root" <<'PY'
import json
import sys
from pathlib import Path

manifests = list(Path(sys.argv[1]).glob("*/manifest.json"))
raise SystemExit(
    0
    if any(json.loads(path.read_text(encoding="utf-8")).get("status") == "completed" for path in manifests)
    else 1
)
PY
}

run_search() {
  local uuid=$1 seed=$2
  local seed_root=$OUTPUT_ROOT/seed-$seed
  local launcher_log=$OUTPUT_ROOT/seed-$seed.launcher.log
  mkdir -p "$seed_root"
  if completed_run_exists "$seed_root"; then
    printf '[%s] seed=%s already completed; skipping\n' "$(date -Is)" "$seed" | tee -a "$launcher_log"
    return
  fi
  local resume_args=()
  local state
  state=$(latest_state "$seed_root")
  if [[ -n "$state" ]]; then
    resume_args=(--resume "$state")
  fi
  printf '[%s] seed=%s gpu=%s population=%s resume=%s\n' \
    "$(date -Is)" "$seed" "$uuid" "$POPULATION" "${state:-none}" | tee -a "$launcher_log"
  CUDA_VISIBLE_DEVICES="$uuid" "$PYTHON" -m zcp_test.cli search \
    --space autoformer --proxy az_nas_autoformer --aggregator az_nas_log_rank \
    --population "$POPULATION" --generations 0 --elite-ratio 0.2 \
    --device cuda:0 --input-source random --batch-size 2 --input-size 224 \
    --classes 1000 --dataset imagenet1k --seed "$seed" \
    --output "$seed_root" "${resume_args[@]}" 2>&1 | tee -a "$launcher_log"
}

lane_a_tasks() {
  local uuid=${gpu_array[0]}
  touch "$OUTPUT_ROOT/lane-a.lock-acquired"
  run_search "$uuid" "${SEEDS[0]}" &
  local first=$!
  run_search "$uuid" "${SEEDS[1]}" &
  local second=$!
  wait "$first"
  wait "$second"
}

lane_b_tasks() {
  local uuid=${gpu_array[1]}
  touch "$OUTPUT_ROOT/lane-b.lock-acquired"
  run_search "$uuid" "${SEEDS[2]}"
}

lane_a() {
  local uuid=${gpu_array[0]}
  acceptance_with_gpu_lock "$LOCK_DIR/$uuid.lock" "$LOCK_TIMEOUT" autoformer-search-lane-a lane_a_tasks
}

lane_b() {
  local uuid=${gpu_array[1]}
  acceptance_with_gpu_lock "$LOCK_DIR/$uuid.lock" "$LOCK_TIMEOUT" autoformer-search-lane-b lane_b_tasks
}

children=()
stop_children() {
  for pid in "${children[@]:-}"; do
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
  done
}
on_error() {
  local exit_code=$?
  set +e
  supervisor_event launcher_error "line=$1 exit_code=$exit_code command=$2"
  stop_children
  write_status failed "search lane failed at line $1 with exit code $exit_code; command=$2"
  exit "$exit_code"
}
on_signal() {
  set +e
  supervisor_event launcher_signal "signal=INT_OR_TERM"
  stop_children
  write_status interrupted "search launcher received signal"
  exit 130
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
trap on_signal INT TERM

supervisor_event launcher_started "commit=$commit gpus=$GPU_UUIDS population=$POPULATION"
write_status queued "waiting for per-GPU locks; launch is automatic and does not block the main workflow"
rm -f "$OUTPUT_ROOT/lane-a.lock-acquired" "$OUTPUT_ROOT/lane-b.lock-acquired"
lane_a & children+=("$!")
lane_b & children+=("$!")
while [[ ! -e "$OUTPUT_ROOT/lane-a.lock-acquired" && ! -e "$OUTPUT_ROOT/lane-b.lock-acquired" ]]; do
  kill -0 "${children[0]}" 2>/dev/null || break
  kill -0 "${children[1]}" 2>/dev/null || break
  sleep 1
done
write_status running "at least one GPU lane acquired; three 8000-candidate seeds are assigned 2+1 across two GPUs"
for pid in "${children[@]}"; do
  wait "$pid"
  supervisor_event lane_wait_completed "child_pid=$pid exit_code=0"
done
write_status completed "all three 8000-candidate AutoFormer AZ-NAS searches completed"
supervisor_event launcher_completed "seeds=${SEEDS[*]}"
