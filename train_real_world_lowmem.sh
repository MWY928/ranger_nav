#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR/habitat-lab:$SCRIPT_DIR/habitat-baselines:$SCRIPT_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TOTAL_NUM_STEPS="${TOTAL_NUM_STEPS:-500000}"

python -m habitat_baselines.run \
  --config-name social_nav_v2/falcon_hm3d_real_world_train_lowmem \
  habitat_baselines.total_num_steps="$TOTAL_NUM_STEPS" \
  "$@"
