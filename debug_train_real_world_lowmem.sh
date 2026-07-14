#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR/habitat-lab:$SCRIPT_DIR/habitat-baselines:$SCRIPT_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TOTAL_NUM_STEPS="${TOTAL_NUM_STEPS:-16000000}"

python -m habitat_baselines.run \
  --config-name social_nav_v2/falcon_hm3d_real_world_train_lowmem \
  habitat_baselines.total_num_steps="$TOTAL_NUM_STEPS" \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.checkpoint_folder="evaluation/falcon/hm3d_real_world_lowmem_jaw_depth_only/checkpoints" \
  habitat_baselines.tensorboard_dir="training/falcon/hm3d_real_world_lowmem_jaw_depth_only/tb" \
  habitat_baselines.video_dir="evaluation/falcon/hm3d_real_world_lowmem_jaw_depth_only/video" \
  habitat.simulator.should_setup_semantic_ids=False \
  '~habitat.simulator.agents.agent_0.sim_sensors.head_rgb_sensor' \
  '~habitat.simulator.agents.agent_0.sim_sensors.head_depth_sensor' \
  '~habitat.simulator.agents.agent_0.sim_sensors.arm_rgb_sensor' \
  '~habitat.simulator.agents.agent_0.sim_sensors.arm_depth_sensor' \
  '~habitat.simulator.agents.agent_0.sim_sensors.head_stereo_left_depth_sensor' \
  '~habitat.simulator.agents.agent_0.sim_sensors.head_stereo_right_depth_sensor' \
  '~habitat.simulator.agents.agent_0.sim_sensors.jaw_rgb_sensor' \
  '~habitat.simulator.agents.agent_0.sim_sensors.arm_panoptic_sensor' \
  '~habitat.simulator.agents.agent_0.sim_sensors.jaw_panoptic_sensor' \
  "$@"
