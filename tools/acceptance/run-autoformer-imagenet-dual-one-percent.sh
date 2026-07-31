#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export ZCP_ACCEPTANCE_SPACE=autoformer
export ZCP_TRAINING_CONFIG=configs/training/autoformer_imagenet.yaml
export ZCP_FORMAL_EPOCHS=500
export ZCP_FULL_DATA_EPOCHS=5
export ZCP_ACCEPTANCE_ROOT=${ZCP_ACCEPTANCE_ROOT:-$PROJECT_ROOT/runs/acceptance/autoformer-imagenet}
exec "$PROJECT_ROOT/tools/acceptance/run-imagenet-candidate-dual-one-percent.sh"
