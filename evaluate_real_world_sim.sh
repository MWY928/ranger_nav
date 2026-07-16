#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR/habitat-lab:$SCRIPT_DIR/habitat-baselines:$SCRIPT_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CKPT_PATH="${1:-evaluation/falcon/hm3d_real_world_lowmem_jaw_depth_only/checkpoints/latest.pth}"
TEST_EPISODE_COUNT="${TEST_EPISODE_COUNT:-10}"
NUM_ENVIRONMENTS="${NUM_ENVIRONMENTS:-1}"
RUN_NAME="${RUN_NAME:-hm3d_real_world_lowmem_jaw_depth_only_eval}"
VIDEO_DIR="${VIDEO_DIR:-evaluation/falcon/${RUN_NAME}/video_n_human_draw}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-training/falcon/${RUN_NAME}/tb}"
EVAL_STATE_DIR="${EVAL_STATE_DIR:-evaluation/falcon/${RUN_NAME}/checkpoints}"

if [ ! -e "$CKPT_PATH" ]; then
  echo "Checkpoint not found: $CKPT_PATH" >&2
  echo "Usage: $0 /path/to/checkpoint.pth" >&2
  exit 1
fi

echo "Evaluating checkpoint: $CKPT_PATH"
echo "Test episodes: $TEST_EPISODE_COUNT"
echo "Num environments: $NUM_ENVIRONMENTS"
echo "Video output dir: $VIDEO_DIR"

python -m habitat_baselines.run \
  --config-name social_nav_v2/falcon_hm3d_real_world_aux_eval \
  habitat_baselines.eval_ckpt_path_dir="$CKPT_PATH" \
  habitat_baselines.num_environments="$NUM_ENVIRONMENTS" \
  habitat_baselines.test_episode_count="$TEST_EPISODE_COUNT" \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.checkpoint_folder="$EVAL_STATE_DIR" \
  habitat_baselines.tensorboard_dir="$TENSORBOARD_DIR" \
  habitat_baselines.video_dir="$VIDEO_DIR" \
  habitat_baselines.eval.video_option='["disk"]' \
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
  "${@:2}"
