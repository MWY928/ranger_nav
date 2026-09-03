#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay one Falcon ROS bridge policy input offline.

This script does not step Habitat-Sim. It only loads the same policy/checkpoint
and feeds a saved real-world observation into the policy path. Use it to check
whether bridge inference and offline inference produce the same action
distribution/value for the exact same policy input.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from gym.spaces import Box
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "habitat-baselines"))
sys.path.append(str(REPO_ROOT / "habitat-lab"))
sys.path.append(str(REPO_ROOT))

from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
from habitat_baselines.utils.common import batch_obs


def _extract_actor_critic_state_dict(
    ckpt_obj: Dict,
) -> Dict[str, torch.Tensor]:
    src = ckpt_obj.get("state_dict", ckpt_obj)
    if not isinstance(src, dict):
        raise RuntimeError(
            "Unsupported checkpoint format: state_dict is not a dict."
        )

    out = {}
    for k, v in src.items():
        if not isinstance(k, str):
            continue
        if "actor_critic." in k:
            out[k.split("actor_critic.", 1)[1]] = v
        elif (
            k.startswith("net.")
            or k.startswith("action_distribution.")
            or k.startswith("critic.")
        ):
            out[k] = v
    return out if len(out) > 0 else src


def _select_checkpoint_state_dict(ckpt_obj):
    if (
        isinstance(ckpt_obj, dict)
        and 0 in ckpt_obj
        and "state_dict" in ckpt_obj[0]
    ):
        return ckpt_obj[0]["state_dict"]
    if (
        isinstance(ckpt_obj, dict)
        and "0" in ckpt_obj
        and "state_dict" in ckpt_obj["0"]
    ):
        return ckpt_obj["0"]["state_dict"]
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        return ckpt_obj["state_dict"]
    return ckpt_obj


def _resolve_sample(path: str) -> str:
    if os.path.isdir(path):
        matches = sorted(
            glob.glob(os.path.join(path, "bridge_policy_replay_*.json"))
        )
        if len(matches) == 0:
            raise RuntimeError(
                "No bridge_policy_replay_*.json found in {}".format(path)
            )
        return matches[-1]
    return path


def _load_sample(sample_path: str):
    sample_path = _resolve_sample(sample_path)
    if sample_path.endswith(".json"):
        with open(sample_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        npz_path = meta["obs_npz"]
    else:
        meta = {}
        npz_path = sample_path

    data = np.load(npz_path)
    return sample_path, meta, data


def _build_policy(args, depth_shape, depth_key: str, goal_key: str):
    observation_space = SpaceDict(
        {
            goal_key: Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            ),
            depth_key: Box(
                low=0.0,
                high=1.0,
                shape=tuple(depth_shape),
                dtype=np.float32,
            ),
        }
    )
    policy = PointNavResNetPolicy(
        observation_space=observation_space,
        action_space=Discrete(4),
        hidden_size=args.hidden_size,
        num_recurrent_layers=args.num_recurrent_layers,
        rnn_type=args.rnn_type,
        backbone=args.backbone,
        normalize_visual_inputs=False,
    ).to(args.device)

    ckpt = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    state_dict = _extract_actor_critic_state_dict(
        _select_checkpoint_state_dict(ckpt)
    )
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    return policy, missing, unexpected


def _fmt_probs(probs: Optional[np.ndarray]) -> str:
    if probs is None or probs.size == 0:
        return "N/A"
    labels = ["stop", "forward", "left", "right"]
    return " ".join(
        "{}:{:.6f}".format(
            labels[i] if i < len(labels) else "a{}".format(i), float(p)
        )
        for i, p in enumerate(probs.tolist())
    )


def main():
    parser = argparse.ArgumentParser(
        description="Replay a saved Falcon bridge policy input"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--sample",
        default="./test_modules/test_results/bridge_policy_replay",
        help="Replay json/npz path, or a directory containing bridge_policy_replay_*.json",
    )
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_recurrent_layers", type=int, default=2)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--rnn_type", type=str, default="LSTM")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--zero_state",
        action="store_true",
        help="Ignore saved RNN input state",
    )
    args = parser.parse_args()

    args.device = torch.device(args.device)
    sample_path, meta, data = _load_sample(args.sample)
    depth_key = meta.get("depth_key", "articulated_agent_jaw_depth")
    goal_key = meta.get("goal_key", "pointgoal_with_gps_compass")

    depth = data["depth"].astype(np.float32)
    goal = data["goal"].astype(np.float32)
    obs = {depth_key: depth, goal_key: goal}

    policy, missing, unexpected = _build_policy(
        args, depth.shape, depth_key, goal_key
    )
    batch = batch_obs([obs], device=args.device)

    if args.zero_state:
        hidden = torch.zeros(
            1,
            policy.net.num_recurrent_layers,
            args.hidden_size,
            device=args.device,
        )
        prev_action = torch.zeros(1, 1, dtype=torch.long, device=args.device)
        mask = torch.zeros(1, 1, dtype=torch.bool, device=args.device)
    else:
        hidden = torch.as_tensor(data["hidden_in"], device=args.device)
        prev_action = torch.as_tensor(
            data["prev_action_in"], dtype=torch.long, device=args.device
        )
        mask = torch.as_tensor(
            data["not_done_mask_in"], dtype=torch.bool, device=args.device
        )

    with torch.no_grad():
        features, _, _ = policy.net(batch, hidden, prev_action, mask)
        distribution = policy.action_distribution(features)
        if policy.action_distribution_type == "categorical":
            action = distribution.mode()
            probs = distribution.probs[0].detach().cpu().numpy()
        else:
            action = distribution.mean
            probs = None
        value = policy.critic(features)

    replay_action = int(action[0][0].item())
    replay_value = float(value[0][0].item())
    saved_executed_action = (
        int(data["action"][0]) if "action" in data else meta.get("action_id")
    )
    saved_policy_action = (
        int(data["raw_action"][0])
        if "raw_action" in data
        else meta.get("raw_action_id", saved_executed_action)
    )
    saved_probs = data["probs"] if "probs" in data else np.array([])
    saved_value = (
        float(data["value"][0])
        if "value" in data
        else meta.get("critic_value")
    )
    saved_value_text = (
        "N/A" if saved_value is None else "{:.6f}".format(float(saved_value))
    )

    print("sample: {}".format(sample_path))
    print("checkpoint: {}".format(args.checkpoint))
    print(
        "load_state_dict missing={} unexpected={}".format(
            len(missing), len(unexpected)
        )
    )
    print(
        "depth shape={} min={:.6f} max={:.6f}".format(
            depth.shape, float(depth.min()), float(depth.max())
        )
    )
    print(
        "goal [r, theta]=[{:.6f}, {:.6f}]".format(
            float(goal[0]), float(goal[1])
        )
    )
    print(
        "saved policy action={} executed action={} value={} probs=[{}]".format(
            saved_policy_action,
            saved_executed_action,
            saved_value_text,
            _fmt_probs(saved_probs),
        )
    )
    print(
        "replay mode action={} value={:.6f} probs=[{}]".format(
            replay_action, replay_value, _fmt_probs(probs)
        )
    )
    if saved_probs is not None and saved_probs.size > 0 and probs is not None:
        print(
            "max prob diff={:.9f}".format(
                float(np.max(np.abs(saved_probs - probs)))
            )
        )
    print(
        "policy action match={}".format(saved_policy_action == replay_action)
    )


if __name__ == "__main__":
    main()
