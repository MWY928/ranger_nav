# Real-World Simulation Changes From Original Falcon

Date: 2026-07-17

This note records the simulation/config changes used for real-world adaptation relative to the original Falcon/Habitat setup.

## Entry Scripts

Added or updated real-world entry scripts:

- `train_real_world_lowmem.sh`
- `debug_train_real_world_lowmem.sh`
- `evaluate_real_world_sim.sh`

The scripts set `PYTHONPATH`, low-memory CUDA allocation defaults, checkpoint/video/tensorboard paths, and Hydra overrides for sensor deletion and real-world reward/action switches.

## Low-Memory Training

`falcon_hm3d_real_world_train_lowmem.yaml` inherits `falcon_hm3d_real_world_train` and changes training resource settings:

- `habitat_baselines.num_environments: 4`
- `habitat_baselines.rl.ppo.num_steps: 64`
- `habitat_baselines.rl.ppo.num_mini_batch: 4`
- `habitat_baselines.rl.ddppo.train_encoder: False`

This keeps training feasible on the current server while reusing the real-world task setup.

## Robot Speed Changes

Original Falcon train config uses faster robot actions:

- move forward linear speed: `6.0`
- turn left angular speed: `6.0`
- turn right angular speed: `-6.0`

Real-world train config changes robot actions to:

- move forward linear speed: `1.0`
- turn left angular speed: `1.0`
- turn right angular speed: `-1.0`

Real-world train also overrides human oracle actions to match the robot speed:

- human linear speed: `1.0`
- human angular speed: `1.0`

Real-world eval config changes robot actions to:

- move forward linear speed: `2.0`
- turn left angular speed: `2.0`
- turn right angular speed: `-2.0`

Real-world eval also overrides human oracle actions to match the eval robot speed:

- human linear speed: `2.0`
- human angular speed: `2.0`

## Robot Body / Agent Setup

The real-world HM3D task uses Spot as `agent_0` with:

- `agent_0.radius: 0.5`
- `kinematic_mode: True`
- `enable_physics: True`
- `allow_sliding: True`

This larger body radius is one likely contributor to conservative behavior or getting stuck in tighter areas.

## Depth Sensor Changes

The real-world task keeps a jaw depth camera as the main visual input:

- resolution: `256 x 256`
- position: `[0.0, 1.25, 0.0]`
- hfov: `90`
- max depth: `10.0`
- `normalize_depth: true`

Realistic depth noise is enabled in the real-world task configs:

- train task:
  - `realistic_noise_enabled: true`
  - `realistic_noise_alpha: 0.0025`
  - `realistic_noise_dropout_base: 0.015`
  - `realistic_noise_dropout_far: 0.083`
  - `realistic_noise_corr_scale: 8`
  - `realistic_noise_edge_dropout: 0.15`
  - `realistic_noise_edge_threshold: 0.3`

- eval task:
  - `realistic_noise_enabled: true`
  - `realistic_noise_alpha: 0.002`
  - `realistic_noise_dropout_base: 0.01`
  - `realistic_noise_dropout_far: 0.08`
  - `realistic_noise_corr_scale: 8`
  - `realistic_noise_edge_dropout: 0.15`
  - `realistic_noise_edge_threshold: 0.3`

## Pointgoal Noise

Real-world train/eval configs replace the clean pointgoal sensor with `noisy_pointgoal_with_gps_compass_sensor`.

Train noise:

- `sigma_r: 0.04`
- `sigma_theta: 0.032`
- `drift_r_std: 0.0015`
- `drift_theta_std: 0.0012`

Eval noise:

- `sigma_r: 0.03`
- `sigma_theta: 0.03490658503988659`
- `drift_r_std: 0.002`
- `drift_theta_std: 0.0017453292519943296`

## Episode Length

Real-world task episode limits:

- train task: `max_episode_steps: 1000`
- eval task: `max_episode_steps: 500`

## Semantic / Panoptic Sensor Handling

The server previously reported missing `.scn` files. Those files are generally needed for semantic scene setup, not pure depth-only navigation.

For real-world low-memory/debug/eval runs, the scripts explicitly disable or remove unnecessary high-memory / semantic-related sensors:

- `habitat.simulator.should_setup_semantic_ids=False` in debug/eval scripts.
- remove RGB/depth sensors that are not used by the policy in debug/eval.
- remove panoptic sensors:
  - `arm_panoptic_sensor`
  - `jaw_panoptic_sensor`

Important distinction:

- removing simulator semantic/panoptic sensors is intended to avoid `.scn` dependency and memory cost.
- removing task lab sensors such as `localization_sensor` is not safe for Falcon social navigation reward, because reward/metrics use robot and human localization internally.

## Observation Filtering

`habitat.gym.obs_keys` controls what the policy receives after the environment computes sensors, reward, and metrics.

Removing a key from `obs_keys` does not necessarily remove the underlying task lab sensor from the environment. This distinction matters:

- OK: keep `localization_sensor` as a lab sensor for reward/metrics, but omit it from policy obs if desired.
- Risky: delete `localization_sensor` from `habitat.task.lab_sensors`, because `MultiAgentNavReward` uses it.

## Evaluation Diagnostics

Added eval diagnostic trace outputs:

- config trace:
  - pretrained weights
  - checkpoint folder
  - resume config flag
  - eval checkpoint path
  - STOP/pause mode switches

- step trace:
  - last action
  - action probabilities
  - robot position delta
  - distance to goal
  - distance reward
  - reward
  - collision metrics

- done trace:
  - terminal action
  - STOP/pause flag
  - distance to goal at done
  - reward at done
  - human collision
  - success/SPL

This was used to confirm that the robot moves, geodesic distance updates, and early failures were dominated by `act_id=0`.

## Real-World Pause Mode Switch

Added a switch to keep the 4-action policy head while changing `act_id=0` from terminal STOP to non-terminal pause/wait.

Default original behavior:

```bash
./train_real_world_lowmem.sh
```

Real-world pause behavior:

```bash
REAL_WORLD_PAUSE_MODE=true ./train_real_world_lowmem.sh
```

When enabled:

- `agent_0_discrete_stop.stop_ends_episode=false`
- `success.auto_success_on_reach=true`
- `multi_agent_nav_reward.pause_mode_enabled=true`
- `multi_agent_nav_reward.no_progress_enabled=true`

This makes the 4 actions behave as:

- `0`: pause/wait
- `1`: move forward
- `2`: turn left
- `3`: turn right

Success is based on reaching `success_distance`, and failure can be triggered by long pause or long no-progress windows.

## Current Open Questions

- Whether the matched robot/human speed should be increased together after the first controlled run.
- Whether robot radius `0.5` is too conservative for the deployed body footprint.
- Whether train/eval observation keys should be aligned more strictly.
- Whether final deployment should move to a 5-action policy with separate terminal DONE and non-terminal PAUSE.
