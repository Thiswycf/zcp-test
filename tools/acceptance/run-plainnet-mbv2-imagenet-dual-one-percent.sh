#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export ZCP_ACCEPTANCE_SPACE=zennas_plainnet_mbv2
export ZCP_TRAINING_CONFIG=configs/training/zennas_plainnet_mbv2_imagenet.yaml
export ZCP_FORMAL_EPOCHS=150
export ZCP_FULL_DATA_EPOCHS=2
export ZCP_ACCEPTANCE_ROOT=${ZCP_ACCEPTANCE_ROOT:-$PROJECT_ROOT/runs/acceptance/plainnet-mbv2-imagenet}
exec "$PROJECT_ROOT/tools/acceptance/run-imagenet-candidate-dual-one-percent.sh"
