#!/usr/bin/env bash
# =========================================================
# run_eval.sh
#
# Purpose:
#   This script is used as a sanity check for the FALCON/Habitat pipeline.
#
#   1. Test whether the Habitat simulation can run successfully.
#   2. Test whether the action outputs are consistent when both the
#      real-robot observation replay and the simulation use fake depth.
#
# Notes:
#   - The checkpoint used for evaluation is:
#       weight/ours_hm3d_val_best.pth
#   - real_obs_replay is enabled.
#   - The replay source is set to "synthetic", meaning that the replayed
#     observation uses manually specified fake depth and fake goal values.
#   - This is mainly for debugging input/action consistency, not for final
#     benchmark evaluation.
# =========================================================
set -e

python -m habitat_baselines.run \
  --config-name social_nav_v2/falcon_hm3d \
  habitat_baselines.eval_ckpt_path_dir=weights/ours_hm3d_val_best.pth \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat.environment.max_episode_steps=20 \
  habitat_baselines.eval.real_obs_replay_enabled=True \
  habitat_baselines.eval.real_obs_replay_source=synthetic \
  habitat_baselines.eval.real_obs_replay_synthetic_depth_m=10.0 \
  habitat_baselines.eval.real_obs_replay_synthetic_goal_r=4.0 \
  habitat_baselines.eval.real_obs_replay_synthetic_goal_theta=0.0\
  habitat_baselines.eval.video_option=["disk"]\
