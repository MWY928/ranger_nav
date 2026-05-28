#!/usr/bin/env bash

# =========================================================
# replay_real_world_trajectory_in_sim.sh
#
# Purpose:
#   This script replays recorded real-world observations in the
#   Habitat/FALCON evaluation pipeline.
#
#   It is mainly used to:
#     1. Test whether the Habitat simulation and evaluation pipeline
#        can run successfully with recorded replay data.
#     2. Check whether the policy action outputs are consistent when
#        using observations recorded from the real-robot bridge.
#     3. Save the replay/evaluation video to disk for visual inspection.
#
# Notes:
#   - The checkpoint used for evaluation is:
#       weights/ours_hm3d_val_best.pth
#   - The replay data path is:
#       test_modules/test_results/bridge_policy_replay
#   - Only one episode is evaluated.
#   - The maximum episode length defaults to the replay sample count.
#   - Video output is enabled and saved to disk.
# =========================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPLAY_ROOT="$SCRIPT_DIR/test_modules/test_results/bridge_policy_replay"

if [ "${1:-}" != "" ]; then
  if [ -d "$1" ]; then
    REPLAY_PATH="$1"
  else
    REPLAY_PATH="$REPLAY_ROOT/$1"
  fi
elif [ ! -d "$REPLAY_ROOT" ]; then
  REPLAY_PATH=""
elif compgen -G "$REPLAY_ROOT/bridge_policy_replay_*.json" > /dev/null; then
  REPLAY_PATH="$REPLAY_ROOT"
else
  REPLAY_PATH="$(find "$REPLAY_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
fi

if [ -z "${REPLAY_PATH:-}" ] || ! compgen -G "$REPLAY_PATH/bridge_policy_replay_*.json" > /dev/null; then
  echo "No replay samples found under: ${REPLAY_PATH:-$REPLAY_ROOT}" >&2
  echo "Expected files like bridge_policy_replay_*.json in a replay run directory." >&2
  exit 1
fi

REPLAY_SAMPLE_COUNT="$(find "$REPLAY_PATH" -maxdepth 1 -type f -name 'bridge_policy_replay_*.json' | wc -l | tr -d ' ')"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-$REPLAY_SAMPLE_COUNT}"

echo "Replaying samples from: $REPLAY_PATH"
echo "Replay sample count: $REPLAY_SAMPLE_COUNT"
echo "Max episode steps: $MAX_EPISODE_STEPS"

python -m habitat_baselines.run \
  --config-name social_nav_v2/falcon_hm3d \
  habitat_baselines.eval_ckpt_path_dir=weights/ours_hm3d_val_best.pth \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat.environment.max_episode_steps="$MAX_EPISODE_STEPS" \
  habitat_baselines.eval.real_obs_replay_enabled=True \
  habitat_baselines.eval.real_obs_replay_path="$REPLAY_PATH" \
  habitat_baselines.eval.video_option='["disk"]'
