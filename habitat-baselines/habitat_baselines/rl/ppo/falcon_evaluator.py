import os
import time
import glob
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm

from habitat import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import Evaluator, pause_envs
from habitat_baselines.rl.multi_agent.utils import update_dict_with_agent_prefix
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info

import json

class FALCONEvaluator(Evaluator):
    """
    Only difference is record the success rate of each episode while evaluating.
    Similar to ORCAEvaluator.
    """

    AGENT0_DISCRETE_ACTION_NAMES = [
        "agent_0_discrete_stop",
        "agent_0_discrete_move_forward",
        "agent_0_discrete_turn_left",
        "agent_0_discrete_turn_right",
    ]

    @staticmethod
    def _depth_stats(arr: np.ndarray) -> Dict[str, float]:
        arr_f = arr.astype(np.float32, copy=False)
        finite = np.isfinite(arr_f)
        valid = arr_f[finite]
        total = float(arr_f.size)
        if valid.size == 0:
            return {
                "valid_ratio": 0.0,
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "p50": float("nan"),
                "p95": float("nan"),
                "zero_ratio": 0.0,
            }

        return {
            "valid_ratio": float(valid.size) / total,
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "mean": float(np.mean(valid)),
            "p50": float(np.percentile(valid, 50)),
            "p95": float(np.percentile(valid, 95)),
            "zero_ratio": float(np.mean(valid == 0.0)),
        }

    @staticmethod
    def _select_depth_key(obs0: Dict[str, Any], config) -> Optional[str]:
        explicit = config.habitat_baselines.eval.depth_dump_obs_key
        if explicit and explicit in obs0:
            return explicit

        for k in config.habitat.gym.obs_keys:
            if "depth" in k and k in obs0:
                return k

        for k in obs0.keys():
            if "depth" in k:
                return k
        return None

    def _dump_depth_sample_once(self, observations, config) -> None:
        if getattr(self, "_depth_sample_dumped", False):
            return
        if not config.habitat_baselines.eval.depth_dump_enabled:
            return
        if observations is None or len(observations) == 0:
            return

        obs0 = observations[0]
        if not isinstance(obs0, dict):
            return

        key = self._select_depth_key(obs0, config)
        if key is None:
            logger.warn("Depth dump enabled but no depth observation key found.")
            return

        depth = obs0[key]
        if torch.is_tensor(depth):
            depth = depth.detach().cpu().numpy()
        depth_np = np.asarray(depth)

        out_dir = config.habitat_baselines.eval.depth_dump_dir
        os.makedirs(out_dir, exist_ok=True)

        stamp = f"{time.time_ns()}_{os.getpid()}"
        safe_key = key.replace("/", "_")
        prefix = f"sim_depth_sample_{safe_key}_{stamp}"

        raw_npy = os.path.join(out_dir, prefix + "_raw.npy")
        raw_csv = os.path.join(out_dir, prefix + "_raw.csv")
        meta_json = os.path.join(out_dir, prefix + "_meta.json")

        np.save(raw_npy, depth_np)
        if config.habitat_baselines.eval.depth_dump_save_csv:
            csv_view = depth_np
            if csv_view.ndim == 3 and csv_view.shape[-1] == 1:
                csv_view = csv_view[..., 0]
            if csv_view.ndim == 2:
                np.savetxt(raw_csv, csv_view, delimiter=",", fmt="%.6f")

        meta = {
            "obs_key": key,
            "shape": list(depth_np.shape),
            "dtype": str(depth_np.dtype),
            "stats": self._depth_stats(depth_np),
            "raw_npy": raw_npy,
            "raw_csv": raw_csv if os.path.exists(raw_csv) else None,
        }
        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        self._depth_sample_dumped = True
        logger.info(
            f"[DepthDump] Saved one depth sample: key={key}, npy={raw_npy}, meta={meta_json}"
        )

    @staticmethod
    def _resolve_real_obs_replay_files(path: str) -> List[str]:
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "bridge_policy_replay_*.json")))
            if len(files) == 0:
                files = sorted(glob.glob(os.path.join(path, "*.npz")))
            return files
        return [path]

    def _load_real_obs_replay_samples(self, config) -> None:
        self._real_obs_replay_samples = []
        self._real_obs_replay_step = 0
        self._real_obs_replay_last = None
        if not config.habitat_baselines.eval.real_obs_replay_enabled:
            return
        if config.habitat_baselines.eval.real_obs_replay_source == "synthetic":
            logger.info("[RealObsReplay] Using synthetic constant depth/goal samples.")
            return
        if config.habitat_baselines.eval.real_obs_replay_source != "file":
            raise RuntimeError(
                "Unsupported real_obs_replay_source: {}".format(
                    config.habitat_baselines.eval.real_obs_replay_source
                )
            )

        files = self._resolve_real_obs_replay_files(
            config.habitat_baselines.eval.real_obs_replay_path
        )
        if len(files) == 0:
            raise RuntimeError(
                "real_obs_replay_enabled=True but no replay samples found in {}".format(
                    config.habitat_baselines.eval.real_obs_replay_path
                )
            )

        for sample_file in files:
            meta = {}
            npz_path = sample_file
            if sample_file.endswith(".json"):
                with open(sample_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                npz_path = meta["obs_npz"]

            data = np.load(npz_path)
            self._real_obs_replay_samples.append(
                {
                    "path": sample_file,
                    "meta": meta,
                    "depth": data["depth"].astype(np.float32)
                    if "depth" in data
                    else None,
                    "depth_meter": data["depth_meter"].astype(np.float32)
                    if "depth_meter" in data
                    else None,
                    "goal": data["goal"].astype(np.float32),
                    "bridge_action": int(data["action"][0])
                    if "action" in data
                    else meta.get("action_id"),
                    "bridge_probs": data["probs"].astype(np.float32).tolist()
                    if "probs" in data and data["probs"].size > 0
                    else meta.get("action_probs"),
                    "bridge_value": float(data["value"][0])
                    if "value" in data
                    else meta.get("critic_value"),
                }
            )

        logger.info(
            "[RealObsReplay] Loaded {} real observation samples from {}".format(
                len(self._real_obs_replay_samples),
                config.habitat_baselines.eval.real_obs_replay_path,
            )
        )

    def _real_obs_replay_depth(self, sample: Dict[str, Any], config) -> np.ndarray:
        unit = config.habitat_baselines.eval.real_obs_replay_depth_unit
        if unit == "meter":
            depth_m = sample["depth_meter"]
            if depth_m is None:
                if sample["depth"] is None:
                    raise RuntimeError("Replay sample has no depth array.")
                depth_m = sample["depth"] * float(
                    config.habitat_baselines.eval.real_obs_replay_max_depth_m
                )
            min_d = float(config.habitat_baselines.eval.real_obs_replay_min_depth_m)
            max_d = float(config.habitat_baselines.eval.real_obs_replay_max_depth_m)
            depth = np.clip(depth_m, min_d, max_d)
            depth = (depth - min_d) / (max_d - min_d)
            return depth.astype(np.float32)
        if unit == "normalized":
            if sample["depth"] is None:
                raise RuntimeError("Replay sample has no normalized depth array.")
            return sample["depth"].astype(np.float32)
        raise RuntimeError("Unsupported real_obs_replay_depth_unit: {}".format(unit))

    def _apply_real_obs_replay(self, observations, config):
        if not config.habitat_baselines.eval.real_obs_replay_enabled:
            return observations
        if config.habitat_baselines.eval.real_obs_replay_source == "synthetic":
            sample = self._make_synthetic_real_obs_sample(observations, config)
            idx = self._real_obs_replay_step
        elif len(self._real_obs_replay_samples) == 0:
            return observations
        else:
            idx = self._real_obs_replay_step
            if idx >= len(self._real_obs_replay_samples):
                if config.habitat_baselines.eval.real_obs_replay_loop:
                    idx = idx % len(self._real_obs_replay_samples)
                else:
                    idx = len(self._real_obs_replay_samples) - 1
            sample = self._real_obs_replay_samples[idx]
        self._real_obs_replay_last = {"sample_index": int(idx), "sample": sample}

        depth_key = config.habitat_baselines.eval.real_obs_replay_depth_key
        if not depth_key:
            depth_key = self._select_depth_key(observations[0], config)
        goal_key = config.habitat_baselines.eval.real_obs_replay_goal_key

        real_depth = self._real_obs_replay_depth(sample, config)
        real_goal = sample["goal"].astype(np.float32)

        for obs in observations:
            if depth_key is not None and depth_key in obs:
                obs[depth_key] = real_depth.copy()
            if goal_key and goal_key in obs:
                obs[goal_key] = real_goal.copy()

        self._real_obs_replay_step += 1
        logger.info(
            "[RealObsReplay] step={} sample={} depth_key={} goal_key={} "
            "goal=[{:.3f}, {:.3f}] depth[min,max]=[{:.3f}, {:.3f}]".format(
                self._real_obs_replay_step,
                sample["path"],
                depth_key,
                goal_key,
                float(real_goal[0]),
                float(real_goal[1]),
                float(np.min(real_depth)),
                float(np.max(real_depth)),
            )
        )
        return observations

    def _make_synthetic_real_obs_sample(self, observations, config) -> Dict[str, Any]:
        depth_key = config.habitat_baselines.eval.real_obs_replay_depth_key
        if not depth_key:
            depth_key = self._select_depth_key(observations[0], config)
        if depth_key is None or depth_key not in observations[0]:
            raise RuntimeError("Cannot create synthetic replay depth: no depth key found.")

        depth_shape = observations[0][depth_key].shape
        min_d = float(config.habitat_baselines.eval.real_obs_replay_min_depth_m)
        max_d = float(config.habitat_baselines.eval.real_obs_replay_max_depth_m)
        depth_m_value = float(config.habitat_baselines.eval.real_obs_replay_synthetic_depth_m)
        depth_norm_value = (np.clip(depth_m_value, min_d, max_d) - min_d) / (max_d - min_d)
        depth = np.full(depth_shape, depth_norm_value, dtype=np.float32)
        depth_meter = np.full(depth_shape, depth_m_value, dtype=np.float32)
        goal = np.array(
            [
                float(config.habitat_baselines.eval.real_obs_replay_synthetic_goal_r),
                float(config.habitat_baselines.eval.real_obs_replay_synthetic_goal_theta),
            ],
            dtype=np.float32,
        )
        return {
            "path": "synthetic_constant",
            "meta": {},
            "depth": depth,
            "depth_meter": depth_meter,
            "goal": goal,
            "bridge_action": None,
            "bridge_probs": None,
            "bridge_value": None,
        }

    @staticmethod
    def _action_to_jsonable(action_value) -> Dict[str, Any]:
        arr = np.asarray(action_value)
        return {
            "type": "array",
            "value": arr.tolist(),
            "scalar": int(arr.reshape(-1)[0]) if arr.size > 0 else None,
        }

    def _record_real_obs_replay_actions(
        self,
        records: List[Dict[str, Any]],
        current_episodes_info,
        step_data,
        sim_probs: Optional[List[List[float]]] = None,
    ) -> None:
        if self._real_obs_replay_last is None:
            return
        sample = self._real_obs_replay_last["sample"]
        sample_index = self._real_obs_replay_last["sample_index"]

        for env_i, action_value in enumerate(step_data):
            action_json = self._action_to_jsonable(action_value)
            env_sim_probs = (
                sim_probs[env_i]
                if sim_probs is not None and env_i < len(sim_probs)
                else None
            )
            records.append(
                {
                    "step": int(self._real_obs_replay_step),
                    "env_index": int(env_i),
                    "scene_id": current_episodes_info[env_i].scene_id,
                    "episode_id": current_episodes_info[env_i].episode_id,
                    "sample_index": int(sample_index),
                    "sample_path": sample["path"],
                    "goal": sample["goal"].astype(np.float32).tolist(),
                    "sim_action": action_json,
                    "sim_probs": env_sim_probs,
                }
            )
            print(
                "[SimAct] step={} env={} action={} probs={}".format(
                    self._real_obs_replay_step, env_i, action_json["scalar"], env_sim_probs
                ),
                flush=True,
            )

    def _write_real_obs_replay_action_records(self, records, config) -> None:
        if not config.habitat_baselines.eval.real_obs_replay_enabled:
            return
        out_path = config.habitat_baselines.eval.real_obs_replay_action_output
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        logger.info(
            "[RealObsReplay] Saved {} action comparison records to {}".format(
                len(records), out_path
            )
        )

    def _compute_agent0_action_probs(
        self,
        agent,
        batch,
        recurrent_hidden_states,
        prev_actions,
        masks,
        hidden_state_lens,
        action_space_lens,
    ) -> Optional[List[List[float]]]:
        try:
            if hasattr(agent, "_agents"):
                policy = agent._agents[0].actor_critic
                obs = update_dict_with_agent_prefix(batch, 0)
                hidden = recurrent_hidden_states[..., : hidden_state_lens[0]]
                prev = prev_actions[..., : action_space_lens[0]]
                mask = masks[..., :1]
            else:
                policy = agent.actor_critic
                obs = batch
                hidden = recurrent_hidden_states
                prev = prev_actions
                mask = masks

            if getattr(policy, "action_distribution_type", None) != "categorical":
                return None

            with torch.no_grad():
                features, _, _ = policy.net(obs, hidden, prev, mask)
                distribution = policy.action_distribution(features)
                return distribution.probs.detach().cpu().numpy().tolist()
        except Exception as e:
            logger.warn("[RealObsReplay] Failed to compute sim action probs: {}".format(e))
            return None

    @classmethod
    def _json_ready(cls, value):
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.numel() == 1:
                return cls._json_ready(value.item())
            return cls._json_ready(value.numpy())
        if isinstance(value, np.ndarray):
            return cls._json_ready(value.tolist())
        if isinstance(value, np.generic):
            return cls._json_ready(value.item())
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): cls._json_ready(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_ready(v) for v in value]
        return str(value)

    @classmethod
    def _float_or_none(cls, value) -> Optional[float]:
        if value is None:
            return None
        try:
            arr = np.asarray(cls._json_ready(value), dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if arr.size == 0 or not np.isfinite(arr[0]):
            return None
        return float(arr[0])

    @classmethod
    def _info_float(cls, info: Dict[str, Any], key: str) -> Optional[float]:
        return cls._float_or_none(info.get(key))

    @classmethod
    def _extract_robot_position(
        cls,
        observation: Dict[str, Any],
        key: str = "agent_0_localization_sensor",
    ) -> Optional[np.ndarray]:
        if not isinstance(observation, dict) or key not in observation:
            return None
        try:
            pos = np.asarray(observation[key], dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if pos.size < 3:
            return None
        return pos[:3].copy()

    @classmethod
    def _robot_position_delta(
        cls,
        prev_pos: Optional[np.ndarray],
        cur_pos: Optional[np.ndarray],
        invalid: bool = False,
    ) -> Dict[str, Any]:
        if invalid:
            return {
                "robot_pos_prev": cls._json_ready(prev_pos),
                "robot_pos": cls._json_ready(cur_pos),
                "robot_pos_delta": None,
                "robot_pos_delta_norm": None,
                "robot_pos_delta_valid": False,
            }
        if prev_pos is None or cur_pos is None:
            return {
                "robot_pos_prev": cls._json_ready(prev_pos),
                "robot_pos": cls._json_ready(cur_pos),
                "robot_pos_delta": None,
                "robot_pos_delta_norm": None,
                "robot_pos_delta_valid": False,
            }
        delta = cur_pos - prev_pos
        return {
            "robot_pos_prev": cls._json_ready(prev_pos),
            "robot_pos": cls._json_ready(cur_pos),
            "robot_pos_delta": cls._json_ready(delta),
            "robot_pos_delta_norm": float(np.linalg.norm(delta)),
            "robot_pos_delta_valid": True,
        }

    @classmethod
    def _extract_robot_position_from_debug_state(
        cls,
        state: Optional[Dict[str, Any]],
    ) -> Optional[np.ndarray]:
        if not isinstance(state, dict):
            return None
        for key in ("articulated_base_pos", "agent_state_position"):
            if key not in state:
                continue
            try:
                pos = np.asarray(state[key], dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                continue
            if pos.size >= 3:
                return pos[:3].copy()
        return None

    def _get_robot_positions(
        self,
        envs,
        observations,
        use_sim_state: bool,
    ) -> Tuple[List[Optional[np.ndarray]], List[Optional[Dict[str, Any]]], str]:
        if use_sim_state and envs.num_envs > 0:
            try:
                states = envs.call(
                    ["get_agent_debug_state"] * envs.num_envs,
                    [{"agent_id": 0} for _ in range(envs.num_envs)],
                )
                return (
                    [
                        self._extract_robot_position_from_debug_state(state)
                        for state in states
                    ],
                    states,
                    "sim_debug_state",
                )
            except Exception as err:
                logger.warn(
                    "[EvalDiagnostics] Failed to read sim debug state: {}".format(
                        err
                    )
                )

        return (
            [self._extract_robot_position(obs) for obs in observations],
            [None for _ in range(len(observations))],
            "observation",
        )

    @classmethod
    def _agent0_action_name(cls, action_id: Optional[int]) -> Optional[str]:
        if action_id is None:
            return None
        if 0 <= action_id < len(cls.AGENT0_DISCRETE_ACTION_NAMES):
            return cls.AGENT0_DISCRETE_ACTION_NAMES[action_id]
        return None

    @classmethod
    def _action_debug(cls, action_value) -> Dict[str, Any]:
        action_json = cls._action_to_jsonable(action_value)
        action_id = action_json["scalar"]
        action_name = cls._agent0_action_name(action_id)
        return {
            "last_action": action_json,
            "agent_0_action_id": action_id,
            "agent_0_action_name": action_name,
            "is_stop_called": action_name == "agent_0_discrete_stop",
        }

    @classmethod
    def _write_jsonl(cls, handle, record: Dict[str, Any], flush: bool) -> None:
        if handle is None:
            return
        handle.write(json.dumps(cls._json_ready(record), ensure_ascii=False) + "\n")
        if flush:
            handle.flush()

    def _config_trace(self, config, checkpoint_index, step_id) -> Dict[str, Any]:
        return {
            "checkpoint_index": checkpoint_index,
            "step_id": step_id,
            "pretrained_weights": config.habitat_baselines.rl.ddppo.pretrained_weights,
            "checkpoint_folder": config.habitat_baselines.checkpoint_folder,
            "load_resume_state_config": config.habitat_baselines.load_resume_state_config,
            "eval_ckpt_path_dir": config.habitat_baselines.eval_ckpt_path_dir,
            "use_ckpt_config": config.habitat_baselines.eval.use_ckpt_config,
            "should_load_ckpt": config.habitat_baselines.eval.should_load_ckpt,
        }

    def _log_config_trace(self, config, checkpoint_index, step_id) -> Dict[str, Any]:
        trace = self._config_trace(config, checkpoint_index, step_id)
        logger.info(
            "[EvalConfigCheck] pretrained_weights={pretrained_weights} "
            "checkpoint_folder={checkpoint_folder} "
            "load_resume_state_config={load_resume_state_config} "
            "eval_ckpt_path_dir={eval_ckpt_path_dir} "
            "use_ckpt_config={use_ckpt_config} "
            "should_load_ckpt={should_load_ckpt}".format(**trace)
        )
        return trace

    def _open_diagnostic_trace_files(self, config, config_trace):
        eval_cfg = config.habitat_baselines.eval
        if not eval_cfg.diagnostic_trace_enabled:
            return None, None

        out_dir = eval_cfg.diagnostic_trace_dir
        os.makedirs(out_dir, exist_ok=True)
        config_path = os.path.join(out_dir, eval_cfg.diagnostic_trace_config_output)
        step_path = os.path.join(out_dir, eval_cfg.diagnostic_trace_step_output)
        done_path = os.path.join(out_dir, eval_cfg.diagnostic_trace_done_output)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self._json_ready(config_trace), f, indent=2, ensure_ascii=False)

        step_handle = open(step_path, "w", encoding="utf-8")
        done_handle = open(done_path, "w", encoding="utf-8")
        logger.info(
            "[EvalDiagnostics] Writing step trace to {} and done trace to {}".format(
                step_path,
                done_path,
            )
        )
        return step_handle, done_handle

    def evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
    ):
        config_trace = self._log_config_trace(config, checkpoint_index, step_id)
        diagnostic_step_handle, diagnostic_done_handle = (
            self._open_diagnostic_trace_files(config, config_trace)
        )
        diagnostic_flush = config.habitat_baselines.eval.diagnostic_trace_flush
        self._depth_sample_dumped = False
        self._load_real_obs_replay_samples(config)
        success_cal = 0 ## my added
        observations = envs.reset()
        observations = envs.post_step(observations)
        observations = self._apply_real_obs_replay(observations, config)
        self._dump_depth_sample_once(observations, config)
        (
            prev_robot_positions,
            _prev_robot_states,
            robot_state_source,
        ) = self._get_robot_positions(
            envs,
            observations,
            diagnostic_step_handle is not None,
        )
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)  

        action_shape, discrete_actions = get_action_space_info(
            agent.actor_critic.policy_action_space
        )

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                config.habitat_baselines.num_environments,
                *agent.actor_critic.hidden_state_shape,
            ),
            device=device,
        )

        hidden_state_lens = agent.actor_critic.hidden_state_shape_lens
        action_space_lens = agent.actor_critic.policy_action_space_shape_lens

        prev_actions = torch.zeros(
            config.habitat_baselines.num_environments,
            *action_shape,
            device=device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            config.habitat_baselines.num_environments,
            *agent.masks_shape,
            device=device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        if len(config.habitat_baselines.eval.video_option) > 0:
            # Add the first frame of the episode to the video.
            rgb_frames: List[List[np.ndarray]] = [
                [
                    observations_to_image(
                        {k: v[env_idx] for k, v in batch.items()}, {}
                    )
                ]
                for env_idx in range(config.habitat_baselines.num_environments)
            ]
        else:
            rgb_frames = None

        if len(config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)
        else:
            total_num_eps = sum(envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        actions_record = defaultdict(list)
        real_obs_replay_action_records = []
        diagnostic_vector_step = 0
        agent.eval()
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            current_episodes_info = envs.current_episodes()

            space_lengths = {}
            n_agents = len(config.habitat.simulator.agents)
            if n_agents > 1:
                space_lengths = {
                    "index_len_recurrent_hidden_states": hidden_state_lens,
                    "index_len_prev_actions": action_space_lens,
                }
            sim_action_probs = None
            if config.habitat_baselines.eval.real_obs_replay_enabled:
                sim_action_probs = self._compute_agent0_action_probs(
                    agent,
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    hidden_state_lens,
                    action_space_lens,
                )
            with inference_mode():
                action_data = agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                    **space_lengths,
                )
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = (
                        action_data.rnn_hidden_states
                    )
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    agent.actor_critic.update_hidden_state(
                        test_recurrent_hidden_states, prev_actions, action_data
                    )

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if hasattr(agent, '_agents') and agent._agents[0]._actor_critic.action_distribution_type == 'categorical':
                step_data = [a.numpy() for a in action_data.env_actions.cpu()]
            elif is_continuous_action_space(env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        env_spec.action_space.low,
                        env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            self._record_real_obs_replay_actions(
                real_obs_replay_action_records,
                current_episodes_info,
                step_data,
                sim_action_probs,
            )

            outputs = envs.step(step_data)
            diagnostic_vector_step += 1

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            for i in range(envs.num_envs):
                episode_key = (
                    current_episodes_info[i].scene_id,
                    current_episodes_info[i].episode_id,
                    ep_eval_count[
                        (current_episodes_info[i].scene_id, current_episodes_info[i].episode_id)
                    ]
                )

                action_value = step_data[i]
                if isinstance(action_value, np.ndarray):
                    stored_action = {
                        "type": "array",
                        "value": action_value.tolist()
                    }
                else:
                    stored_action = {
                        "type": "array",
                        "value": np.array(action_value).tolist()
                    }

                actions_record[episode_key].append(stored_action)

            # Note that `policy_infos` represents the information about the
            # action BEFORE `observations` (the action used to transition to
            # `observations`).
            policy_infos = agent.actor_critic.get_extra(
                action_data, infos, dones
            )
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])

            observations = envs.post_step(observations)
            observations = self._apply_real_obs_replay(observations, config)
            (
                current_robot_positions,
                current_robot_states,
                robot_state_source,
            ) = self._get_robot_positions(
                envs,
                observations,
                diagnostic_step_handle is not None,
            )
            batch = batch_obs(  # type: ignore
                observations,
                device=device,
            )
            batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            ).repeat(1, *agent.masks_shape)

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            for i in range(envs.num_envs):
                episode_key = (
                    current_episodes_info[i].scene_id,
                    current_episodes_info[i].episode_id,
                    ep_eval_count[
                        (current_episodes_info[i].scene_id, current_episodes_info[i].episode_id)
                    ],
                )
                action_debug = self._action_debug(step_data[i])
                step_record = {
                    "event": "step",
                    "vector_step": diagnostic_vector_step,
                    "env_index": i,
                    "scene_id": current_episodes_info[i].scene_id,
                    "episode_id": current_episodes_info[i].episode_id,
                    "episode_eval_index": episode_key[2],
                    "episode_step": len(actions_record[episode_key]),
                    "done": bool(dones[i]),
                    "done_observation_is_auto_reset": bool(dones[i]),
                    "reward": self._float_or_none(rewards_l[i]),
                    "episode_reward_so_far": float(current_episode_reward[i].item()),
                    "distance_to_goal": self._info_float(infos[i], "distance_to_goal"),
                    "distance_to_goal_reward": self._info_float(
                        infos[i], "distance_to_goal_reward"
                    ),
                    "human_collision": self._info_float(infos[i], "human_collision"),
                    "success": self._info_float(infos[i], "success"),
                    "spl": self._info_float(infos[i], "spl"),
                    "num_steps": self._info_float(infos[i], "num_steps"),
                    "robot_state_source": robot_state_source,
                    "robot_debug_state": current_robot_states[i]
                    if i < len(current_robot_states)
                    else None,
                    **action_debug,
                    **self._robot_position_delta(
                        prev_robot_positions[i]
                        if i < len(prev_robot_positions)
                        else None,
                        current_robot_positions[i],
                        invalid=bool(dones[i]),
                    ),
                }
                self._write_jsonl(
                    diagnostic_step_handle,
                    step_record,
                    diagnostic_flush,
                )
            prev_robot_positions = current_robot_positions
            next_episodes_info = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)

                # Exclude the keys from `_rank0_keys` from displaying in the video
                disp_info = {
                    k: v for k, v in infos[i].items() if k not in rank0_keys
                }

                if len(config.habitat_baselines.eval.video_option) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, disp_info
                    )
                    if not not_done_masks[i].any().item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        final_frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()},
                            disp_info,
                        )
                        final_frame = overlay_frame(final_frame, disp_info)
                        rgb_frames[i].append(final_frame)
                        # The starting frame of the next episode will be the final element..
                        rgb_frames[i].append(frame)
                    else:
                        frame = overlay_frame(frame, disp_info)
                        rgb_frames[i].append(frame)

                # episode ended
                if not not_done_masks[i].any().item():
                    pbar.update()
                    if "success" in disp_info:
                        success_cal += disp_info['success']
                        print(f"Till now Success Rate: {success_cal/(len(stats_episodes)+1)}")
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    completed_eval_count = ep_eval_count[k] + 1
                    action_debug = self._action_debug(step_data[i])
                    done_record = {
                        "event": "done",
                        "vector_step": diagnostic_vector_step,
                        "env_index": i,
                        "scene_id": current_episodes_info[i].scene_id,
                        "episode_id": current_episodes_info[i].episode_id,
                        "episode_eval_count_completed": completed_eval_count,
                        "last_episode_step": len(
                            actions_record[
                                (
                                    current_episodes_info[i].scene_id,
                                    current_episodes_info[i].episode_id,
                                    ep_eval_count[k],
                                )
                            ]
                        ),
                        "reward": episode_stats["reward"],
                        "last_step_reward": self._float_or_none(rewards_l[i]),
                        "distance_to_goal": self._info_float(infos[i], "distance_to_goal"),
                        "distance_to_goal_reward": self._info_float(
                            infos[i], "distance_to_goal_reward"
                        ),
                        "human_collision": self._info_float(infos[i], "human_collision"),
                        "success": self._info_float(infos[i], "success"),
                        "spl": self._info_float(infos[i], "spl"),
                        "num_steps": self._info_float(infos[i], "num_steps"),
                        **action_debug,
                    }
                    self._write_jsonl(
                        diagnostic_done_handle,
                        done_record,
                        diagnostic_flush,
                    )
                    current_episode_reward[i] = 0
                    ep_eval_count[k] = completed_eval_count
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if len(config.habitat_baselines.eval.video_option) > 0:
                        # show scene and episode
                        scene_id = current_episodes_info[i].scene_id.split('/')[-1].split('.')[0]
                        print(f"This is Scene ID: {scene_id}, Episode ID: {current_episodes_info[i].episode_id}.") # for debug
                        
                        generate_video(
                            video_option=config.habitat_baselines.eval.video_option,
                            video_dir=config.habitat_baselines.video_dir,
                            # Since the final frame is the start frame of the next episode.
                            images=rgb_frames[i][:-1],
                            scene_id=f"{current_episodes_info[i].scene_id}".split('/')[-1].split('.')[0],
                            episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(disp_info),
                            fps=config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        # Since the starting frame of the next episode is the final frame.
                        rgb_frames[i] = rgb_frames[i][-1:]

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=device)
            (
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = pause_envs(
                envs_to_pause,
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )
            if any(envs_to_pause):
                paused = set(envs_to_pause)
                prev_robot_positions = [
                    pos
                    for env_idx, pos in enumerate(prev_robot_positions)
                    if env_idx not in paused
                ]

            # We pause the statefull parameters in the policy.
            # We only do this if there are envs to pause to reduce the overhead.
            # In addition, HRL policy requires the solution_actions to be non-empty, and
            # empty list of envs_to_pause will raise an error.
            if any(envs_to_pause):
                agent.actor_critic.on_envs_pause(envs_to_pause)

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        all_ks = set()
        for ep in stats_episodes.values():
            all_ks.update(ep.keys())
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        # ==== 保存 result.json ====
        result_path = os.path.join("output/", "result.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        evalai_result = {
                            "SR": round(aggregated_stats.get("success", 0), 4),
                            "SPL": round(aggregated_stats.get("spl", 0), 4),
                            "PSC": round(aggregated_stats.get("psc", 0), 4),
                            "H-Coll": round(aggregated_stats.get("human_collision", 0), 4),
                            "Total": round(
                                0.4 * aggregated_stats.get("success", 0)
                                + 0.3 * aggregated_stats.get("spl", 0)
                                + 0.3 * aggregated_stats.get("psc", 0),
                                4,
                                    ),
                        }

        with open(result_path, "w") as f:
            json.dump(evalai_result, f, indent=2)

        # ==== 保存 actions.json ====
        actions_output_path = os.path.join("output/", "actions.json")
        os.makedirs(os.path.dirname(actions_output_path), exist_ok=True)
        serializable_actions = {
            f"{scene_id}|{episode_id}|{eval_count}": actions
            for (scene_id, episode_id, eval_count), actions in actions_record.items()
        }
        with open(actions_output_path, "w") as f:
            json.dump(serializable_actions, f, indent=2)

        self._write_real_obs_replay_action_records(
            real_obs_replay_action_records,
            config,
        )
        for diagnostic_handle in (diagnostic_step_handle, diagnostic_done_handle):
            if diagnostic_handle is not None:
                diagnostic_handle.close()
