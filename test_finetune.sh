#!/bin/bash

set -e

# =========================
# Paths
# =========================
CHECKPOINT="weights/ours_hm3d_val_best.pth"

DATA_ROOT="./test_modules/test_results/il_trajectories"

OUTPUT_CKPT="./checkpoints/falcon_bc_70traj_action_head_lstm2.pth"

HABITAT_BASELINES="/home/mobile/ranger_nav/habitat-baselines"
HABITAT_LAB="/home/mobile/ranger_nav/habitat-lab"
PROJECT_ROOT="/home/mobile/ranger_nav"

mkdir -p "$(dirname "${OUTPUT_CKPT}")"

# =========================
# BC fine-tuning on 70 trajectories
# =========================
python IL_SFT/offline_bc_finetune.py \
    --checkpoint "${CHECKPOINT}" \
    --data_root "${DATA_ROOT}" \
    --output_checkpoint "${OUTPUT_CKPT}" \
    --habitat_baselines_path "${HABITAT_BASELINES}" \
    --habitat_lab_path "${HABITAT_LAB}" \
    --project_path "${PROJECT_ROOT}" \
    --epochs 35 \
    --batch_size 4 \
    --freeze_mode action_head_lstm \
    --lr 1e-4 \
    --class_balance \
    --max_consecutive_stop 6

echo ""
echo "===================================="
echo "trajectory BC fine-tuning finished."
echo "Checkpoint saved to:"
echo "${OUTPUT_CKPT}"
echo "===================================="
