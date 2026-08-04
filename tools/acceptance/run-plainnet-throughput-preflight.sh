#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${ZCP_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
OUTPUT_ROOT=${ZCP_PLAINNET_PREFLIGHT_ROOT:-${ZCP_ACCEPTANCE_ROOT:-$PROJECT_ROOT/runs/acceptance/plainnet-source-aligned-throughput-preflight}}
GPU_UUID=${ZCP_PLAINNET_PREFLIGHT_GPU_UUID:?set ZCP_PLAINNET_PREFLIGHT_GPU_UUID to a GPU UUID}
LOCK_TIMEOUT=${ZCP_GPU_LOCK_TIMEOUT_SECONDS:-21600}
PYTHON=${ZCP_PYTHON:-$(command -v python)}

source "$PROJECT_ROOT/tools/acceptance/lib/launcher-runtime.sh"
acceptance_exec_immutable "$PROJECT_ROOT" "$OUTPUT_ROOT" "${BASH_SOURCE[0]}" "$@"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TZ=Asia/Shanghai
mkdir -p "$OUTPUT_ROOT"

"$PYTHON" "$PROJECT_ROOT/tools/acceptance/preflight-plainnet-source-aligned.py" \
  --gpu "$GPU_UUID" \
  --lock-timeout "$LOCK_TIMEOUT" \
  --accepted 3 \
  --flops-target 450m \
  --output "$OUTPUT_ROOT"
