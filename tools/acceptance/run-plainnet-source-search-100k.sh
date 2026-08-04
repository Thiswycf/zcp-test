#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${ZCP_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${ZCP_PYTHON:-python}
GPU_UUID=${ZCP_PLAINNET_SEARCH_GPU_UUID:?Set ZCP_PLAINNET_SEARCH_GPU_UUID to one GPU UUID}
FLOPS_TARGET=${ZCP_PLAINNET_FLOPS_TARGET:?Set ZCP_PLAINNET_FLOPS_TARGET to 450m, 600m, or 1g}
OUTPUT_ROOT=${ZCP_PLAINNET_SEARCH_ROOT:-$PROJECT_ROOT/runs/acceptance/plainnet-source-aligned-100k/$FLOPS_TARGET}
LOCK_TIMEOUT=${ZCP_GPU_LOCK_TIMEOUT_SECONDS:-21600}
STATUS=$OUTPUT_ROOT/status.json
LAUNCHER_LOG=$OUTPUT_ROOT/launcher.log

source "$PROJECT_ROOT/tools/acceptance/lib/launcher-runtime.sh"
acceptance_exec_immutable "$PROJECT_ROOT" "$OUTPUT_ROOT" "${BASH_SOURCE[0]}" "$@"

[[ "$GPU_UUID" =~ ^GPU-[A-Fa-f0-9-]+$ ]] || { echo "Invalid GPU UUID: $GPU_UUID" >&2; exit 2; }
[[ "$FLOPS_TARGET" =~ ^(450m|600m|1g)$ ]] || { echo "Invalid FLOPS target: $FLOPS_TARGET" >&2; exit 2; }
[[ "$LOCK_TIMEOUT" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "GPU lock timeout must be non-negative" >&2; exit 2; }

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TZ=Asia/Shanghai
export PYTHONPATH=$PROJECT_ROOT/src
mkdir -p "$OUTPUT_ROOT"
commit=${ZCP_LAUNCHER_COMMIT:-$(git -C "$PROJECT_ROOT" rev-parse HEAD)}

write_status() {
  local state=$1 detail=$2
  "$PYTHON" - "$STATUS" "$state" "$detail" "$commit" "$GPU_UUID" "$FLOPS_TARGET" <<'PY'
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
    "schema_version": "1.0",
    "status": sys.argv[2],
    "detail": sys.argv[3],
    "project_commit": sys.argv[4],
    "gpu_uuid": sys.argv[5],
    "flops_target": sys.argv[6],
    "valid_candidates": 100000,
    "controller": "plainnet_source_aligned",
    "formal_search_completed": sys.argv[2] == "completed",
    "pid": os.getppid(),
    "started_at": existing.get("started_at", now),
    "updated_at": now,
}
if sys.argv[2] in {"completed", "failed", "interrupted"}:
    payload["ended_at"] = now
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

completed_run_exists() {
  "$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

for manifest in Path(sys.argv[1]).glob("*/manifest.json"):
    if json.loads(manifest.read_text(encoding="utf-8")).get("status") == "completed":
        raise SystemExit(0)
raise SystemExit(1)
PY
}

latest_state() {
  find "$OUTPUT_ROOT" -mindepth 2 -maxdepth 2 -type f -name search-state.json \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
}

if completed_run_exists; then
  write_status completed "a terminal completed run already exists; launcher skipped"
  exit 0
fi

resume_args=()
state=$(latest_state)
if [[ -n "$state" ]]; then
  resume_args=(--resume "$state")
fi

write_status running "formal 100k source-aligned search"
printf '[%s] target=%s gpu=%s resume=%s commit=%s\n' \
  "$(date -Is)" "$FLOPS_TARGET" "$GPU_UUID" "${state:-none}" "$commit" | tee -a "$LAUNCHER_LOG"

set +e
"$PYTHON" -m zcp_test.cli search \
  --config "$PROJECT_ROOT/configs/search/plainnet_mbv2_source_aligned.yaml" \
  --flops-target "$FLOPS_TARGET" \
  --gpu "$GPU_UUID" \
  --gpu-lock-timeout "$LOCK_TIMEOUT" \
  --output "$OUTPUT_ROOT" \
  "${resume_args[@]}" 2>&1 | tee -a "$LAUNCHER_LOG"
exit_code=${PIPESTATUS[0]}
set -e

if [[ $exit_code -eq 0 ]]; then
  write_status completed "formal 100k source-aligned search completed"
elif [[ $exit_code -eq 130 || $exit_code -eq 143 ]]; then
  write_status interrupted "launcher received a termination signal"
else
  write_status failed "search exited with code $exit_code"
fi
exit "$exit_code"
