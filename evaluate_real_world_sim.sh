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
DIAGNOSTIC_TRACE_ENABLED="${DIAGNOSTIC_TRACE_ENABLED:-true}"
DIAGNOSTIC_TRACE_DIR="${DIAGNOSTIC_TRACE_DIR:-output/eval_diagnostics/${RUN_NAME}}"
REAL_WORLD_STOP_REWARD="${REAL_WORLD_STOP_REWARD:-false}"
FAR_STOP_PENALTY="${FAR_STOP_PENALTY:--0.25}"
FAR_STOP_DISTANCE_THRESHOLD="${FAR_STOP_DISTANCE_THRESHOLD:--1.0}"
# This evaluator targets checkpoints trained with REAL_WORLD_PAUSE_MODE=true.
# Set REAL_WORLD_PAUSE_MODE=false explicitly to restore terminal STOP behavior.
REAL_WORLD_PAUSE_MODE="${REAL_WORLD_PAUSE_MODE:-true}"
if [[ "$REAL_WORLD_PAUSE_MODE" == "true" || "$REAL_WORLD_PAUSE_MODE" == "1" ]]; then
  STOP_ENDS_EPISODE="${STOP_ENDS_EPISODE:-false}"
  AUTO_SUCCESS_ON_REACH="${AUTO_SUCCESS_ON_REACH:-true}"
  PAUSE_MODE_ENABLED="${PAUSE_MODE_ENABLED:-true}"
  NO_PROGRESS_ENABLED="${NO_PROGRESS_ENABLED:-true}"
else
  STOP_ENDS_EPISODE="${STOP_ENDS_EPISODE:-true}"
  AUTO_SUCCESS_ON_REACH="${AUTO_SUCCESS_ON_REACH:-false}"
  PAUSE_MODE_ENABLED="${PAUSE_MODE_ENABLED:-false}"
  NO_PROGRESS_ENABLED="${NO_PROGRESS_ENABLED:-false}"
fi
PAUSE_GRACE_STEPS="${PAUSE_GRACE_STEPS:-60}"
PAUSE_MAX_STEPS="${PAUSE_MAX_STEPS:-180}"
PAUSE_STEP_PENALTY="${PAUSE_STEP_PENALTY:--0.002}"
PAUSE_DONE_PENALTY="${PAUSE_DONE_PENALTY:--0.25}"
NO_PROGRESS_WINDOW="${NO_PROGRESS_WINDOW:-120}"
NO_PROGRESS_MOVE_EPS="${NO_PROGRESS_MOVE_EPS:-0.005}"
NO_PROGRESS_DISTANCE_EPS="${NO_PROGRESS_DISTANCE_EPS:-0.01}"
NO_PROGRESS_DONE_PENALTY="${NO_PROGRESS_DONE_PENALTY:--0.25}"

if [ ! -e "$CKPT_PATH" ]; then
  echo "Checkpoint not found: $CKPT_PATH" >&2
  echo "Usage: $0 /path/to/checkpoint.pth [hydra overrides...]" >&2
  exit 1
fi

echo "Evaluating checkpoint: $CKPT_PATH"
echo "Test episodes: $TEST_EPISODE_COUNT"
echo "Num environments: $NUM_ENVIRONMENTS"
echo "Video output dir: $VIDEO_DIR"
echo "Diagnostic trace enabled: $DIAGNOSTIC_TRACE_ENABLED"
echo "Diagnostic trace dir: $DIAGNOSTIC_TRACE_DIR"
echo "Real-world pause mode: $REAL_WORLD_PAUSE_MODE"

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
  habitat_baselines.eval.diagnostic_trace_enabled="$DIAGNOSTIC_TRACE_ENABLED" \
  habitat_baselines.eval.diagnostic_trace_dir="$DIAGNOSTIC_TRACE_DIR" \
  habitat.task.actions.agent_0_discrete_stop.stop_ends_episode="$STOP_ENDS_EPISODE" \
  habitat.task.measurements.success.auto_success_on_reach="$AUTO_SUCCESS_ON_REACH" \
  habitat.task.measurements.multi_agent_nav_reward.far_stop_penalty_enabled="$REAL_WORLD_STOP_REWARD" \
  habitat.task.measurements.multi_agent_nav_reward.far_stop_penalty="$FAR_STOP_PENALTY" \
  habitat.task.measurements.multi_agent_nav_reward.far_stop_distance_threshold="$FAR_STOP_DISTANCE_THRESHOLD" \
  habitat.task.measurements.multi_agent_nav_reward.pause_mode_enabled="$PAUSE_MODE_ENABLED" \
  habitat.task.measurements.multi_agent_nav_reward.pause_grace_steps="$PAUSE_GRACE_STEPS" \
  habitat.task.measurements.multi_agent_nav_reward.pause_max_steps="$PAUSE_MAX_STEPS" \
  habitat.task.measurements.multi_agent_nav_reward.pause_step_penalty="$PAUSE_STEP_PENALTY" \
  habitat.task.measurements.multi_agent_nav_reward.pause_done_penalty="$PAUSE_DONE_PENALTY" \
  habitat.task.measurements.multi_agent_nav_reward.no_progress_enabled="$NO_PROGRESS_ENABLED" \
  habitat.task.measurements.multi_agent_nav_reward.no_progress_window="$NO_PROGRESS_WINDOW" \
  habitat.task.measurements.multi_agent_nav_reward.no_progress_move_eps="$NO_PROGRESS_MOVE_EPS" \
  habitat.task.measurements.multi_agent_nav_reward.no_progress_distance_eps="$NO_PROGRESS_DISTANCE_EPS" \
  habitat.task.measurements.multi_agent_nav_reward.no_progress_done_penalty="$NO_PROGRESS_DONE_PENALTY" \
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
