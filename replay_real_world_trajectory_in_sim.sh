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
#   - The maximum episode length is limited to 60 steps.
#   - Video output is enabled and saved to disk.
# =========================================================

set -e

python -m habitat_baselines.run \
  --config-name social_nav_v2/falcon_hm3d \
  habitat_baselines.eval_ckpt_path_dir=weights/ours_hm3d_val_best.pth \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat.environment.max_episode_steps=60 \
  habitat_baselines.eval.real_obs_replay_enabled=True \
  habitat_baselines.eval.real_obs_replay_path=test_modules/test_results/bridge_policy_replay \
  habitat_baselines.eval.video_option='["disk"]'