#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline behavior-cloning fine-tune script for Falcon / PointNavResNetPolicy.

This script is designed for trajectory data collected by collect_imitation_trajectory.py:

    run_dir/
        run_meta.json
        manifest.jsonl
        samples/
            sample_000000_*.npz
            sample_000001_*.npz
            ...

Each .npz is expected to contain:
    depth:  [H, W] float32, meters
    goal:   [2] float32, [r_m, theta_rad]
    action: [1] int64, Falcon action id

The script does NOT start Habitat-Sim. It only reuses the Habitat/Falcon policy class,
constructs the same observation/action spaces as falcon_ros_bridge.py, and performs
offline supervised fine-tuning on expert trajectories.

Typical usage:

python IL_SFT/offline_bc_finetune.py \
    --checkpoint weights/ours_hm3d_val_best.pth \
    --data_root ./test_modules/test_results/il_trajectories \
    --output_checkpoint ./checkpoints/falcon_real_depth_bc.pth \
    --habitat_baselines_path /home/mobile/ranger_nav/habitat-baselines \
    --habitat_lab_path /home/mobile/ranger_nav/habitat-lab \
    --project_path /home/mobile/ranger_nav \
    --freeze_mode action_head \
    --epochs 20 \
    --batch_size 4 \
    --lr 1e-4 \
    --kl_coef 0.02 \
    --class_balance

Recommended first sanity check:

python offline_bc_finetune.py \
    --checkpoint /path/to/original_falcon_ckpt.pth \
    --data_root ./test_modules/test_results/il_trajectories \
    --output_checkpoint ./checkpoints/overfit_debug.pth \
    --overfit_n_episodes 3 \
    --epochs 50 \
    --batch_size 3 \
    --freeze_mode action_head_lstm \
    --lr 1e-4
"""

import argparse
import copy
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from gym.spaces import Box
    from gym.spaces import Dict as SpaceDict
    from gym.spaces import Discrete
except Exception as exc:  # pragma: no cover - environment-specific
    raise RuntimeError(
        "Could not import gym.spaces. Please run inside the same environment used by Falcon/Habitat."
    ) from exc


ACTION_NAMES = {
    0: "stop",
    1: "forward",
    2: "left",
    3: "right",
}


# -----------------------------------------------------------------------------
# Checkpoint utilities copied/adapted from falcon_ros_bridge.py.
# -----------------------------------------------------------------------------


def safe_torch_load(path: str, map_location):
    """torch.load wrapper compatible with both old and new PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _extract_actor_critic_state_dict(ckpt_obj: Dict) -> Dict[str, torch.Tensor]:
    """
    Accept different checkpoint layouts and return state_dict keys compatible
    with PointNavResNetPolicy.
    """
    src = ckpt_obj.get("state_dict", ckpt_obj) if isinstance(ckpt_obj, dict) else ckpt_obj
    if not isinstance(src, dict):
        raise RuntimeError("Unsupported checkpoint format: state_dict is not a dict.")

    out = {}
    for k, v in src.items():
        if not isinstance(k, str) or not torch.is_tensor(v):
            continue
        if "actor_critic." in k:
            kk = k.split("actor_critic.", 1)[1]
            out[kk] = v
        elif k.startswith("net.") or k.startswith("action_distribution.") or k.startswith("critic."):
            out[k] = v

    if len(out) == 0:
        out = {k: v for k, v in src.items() if isinstance(k, str) and torch.is_tensor(v)}

    if len(out) == 0:
        raise RuntimeError("No string-key tensor parameters found in checkpoint.")
    return out


def _select_agent0_checkpoint_state_dict(ckpt_obj):
    """Return the policy state_dict from common single-agent or multi-agent checkpoint layouts."""
    if isinstance(ckpt_obj, (list, tuple)) and len(ckpt_obj) > 0 and isinstance(ckpt_obj[0], dict):
        return ckpt_obj[0].get("state_dict", ckpt_obj[0])

    if isinstance(ckpt_obj, dict):
        if 0 in ckpt_obj and isinstance(ckpt_obj[0], dict):
            return ckpt_obj[0].get("state_dict", ckpt_obj[0])
        if "0" in ckpt_obj and isinstance(ckpt_obj["0"], dict):
            return ckpt_obj["0"].get("state_dict", ckpt_obj["0"])
        if "state_dict" in ckpt_obj:
            return ckpt_obj["state_dict"]

    return ckpt_obj


# -----------------------------------------------------------------------------
# Import/build policy.
# -----------------------------------------------------------------------------


def add_optional_sys_paths(paths: Sequence[Optional[str]]) -> None:
    for p in paths:
        if p is None or str(p).strip() == "":
            continue
        p = os.path.abspath(os.path.expanduser(str(p)))
        if p not in sys.path:
            sys.path.append(p)


def import_pointnav_policy(args):
    add_optional_sys_paths(
        [
            args.habitat_baselines_path,
            args.habitat_lab_path,
            args.project_path,
        ]
    )
    try:
        from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Could not import PointNavResNetPolicy. Please set --habitat_baselines_path, "
            "--habitat_lab_path, and --project_path to match your Falcon ROS bridge environment."
        ) from exc
    return PointNavResNetPolicy


def build_policy(args, device: torch.device):
    PointNavResNetPolicy = import_pointnav_policy(args)

    spaces = {
        args.goal_obs_key: Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,),
            dtype=np.float32,
        ),
        args.depth_obs_key: Box(
            low=0.0,
            high=1.0,
            shape=(args.resolution, args.resolution, 1),
            dtype=np.float32,
        ),
    }
    observation_space = SpaceDict(spaces)
    action_space = Discrete(4)

    policy = PointNavResNetPolicy(
        observation_space=observation_space,
        action_space=action_space,
        hidden_size=args.hidden_size,
        num_recurrent_layers=args.num_recurrent_layers,
        rnn_type=args.rnn_type,
        backbone=args.backbone,
        normalize_visual_inputs=False,
    ).to(device)

    ckpt = safe_torch_load(args.checkpoint, map_location=device)
    selected = _select_agent0_checkpoint_state_dict(ckpt)
    policy_sd = _extract_actor_critic_state_dict(selected)
    missing, unexpected = policy.load_state_dict(policy_sd, strict=False)

    print("[CKPT] loaded:", args.checkpoint)
    print(f"[CKPT] missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print("[CKPT] first missing keys:", missing[:20])
    if len(unexpected) > 0:
        print("[CKPT] first unexpected keys:", unexpected[:20])
    if args.strict_checkpoint and (len(missing) > 0 or len(unexpected) > 0):
        raise RuntimeError("Checkpoint mismatch under --strict_checkpoint.")

    return policy


# -----------------------------------------------------------------------------
# Dataset.
# -----------------------------------------------------------------------------


def read_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def discover_run_dirs(data_roots: Sequence[str]) -> List[Path]:
    """Find directories containing manifest.jsonl."""
    run_dirs: List[Path] = []
    seen = set()

    for root in data_roots:
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            raise FileNotFoundError(f"Data root does not exist: {root_path}")

        candidates: List[Path]
        if root_path.is_file() and root_path.name == "manifest.jsonl":
            candidates = [root_path.parent]
        elif (root_path / "manifest.jsonl").exists():
            candidates = [root_path]
        else:
            candidates = [p.parent for p in root_path.rglob("manifest.jsonl")]

        for d in candidates:
            d = d.resolve()
            if d not in seen:
                seen.add(d)
                run_dirs.append(d)

    run_dirs.sort(key=lambda p: str(p))
    return run_dirs


def resolve_sample_path(run_dir: Path, sample_npz: str) -> Path:
    p = Path(sample_npz).expanduser()
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()

    # The manifest written by the collector stores absolute paths. If the dataset
    # was copied to another machine, fall back to run_dir/samples/<basename>.
    fallback = run_dir / "samples" / Path(sample_npz).name
    if fallback.exists():
        return fallback.resolve()

    # Last attempt: relative to run_dir.
    fallback = run_dir / sample_npz
    if fallback.exists():
        return fallback.resolve()

    raise FileNotFoundError(f"Sample npz not found: {sample_npz} under {run_dir}")


def trim_repeated_stops(records: List[Dict], max_consecutive_stop: int) -> List[Dict]:
    if max_consecutive_stop < 0:
        return records
    out: List[Dict] = []
    stop_count = 0
    for rec in records:
        action_id = int(rec.get("action_id", -1))
        if action_id == 0:
            stop_count += 1
        else:
            stop_count = 0
        if action_id == 0 and stop_count > max_consecutive_stop:
            continue
        out.append(rec)
    return out


class FalconImitationEpisodeDataset(Dataset):
    """Each item is one trajectory/run_dir."""

    def __init__(
        self,
        run_dirs: Sequence[Path],
        resolution: int,
        max_depth_m: float,
        max_consecutive_stop: int = 3,
        min_episode_len: int = 1,
    ):
        self.run_dirs = list(run_dirs)
        self.resolution = int(resolution)
        self.max_depth_m = float(max_depth_m)
        self.max_consecutive_stop = int(max_consecutive_stop)
        self.min_episode_len = int(min_episode_len)
        self.episodes: List[Dict] = []
        self._load_manifests()

    def _load_manifests(self) -> None:
        for run_dir in self.run_dirs:
            manifest_path = run_dir / "manifest.jsonl"
            if not manifest_path.exists():
                continue
            records = read_jsonl(manifest_path)
            records.sort(key=lambda r: int(r.get("index", 0)))
            records = trim_repeated_stops(records, self.max_consecutive_stop)
            if len(records) < self.min_episode_len:
                continue

            meta = {}
            meta_path = run_dir / "run_meta.json"
            if meta_path.exists():
                try:
                    with meta_path.open("r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    meta = {}

            self.episodes.append(
                {
                    "run_dir": run_dir,
                    "records": records,
                    "meta": meta,
                }
            )

        if len(self.episodes) == 0:
            raise RuntimeError("No valid episodes found. Check --data_root and manifest.jsonl files.")

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        ep = self.episodes[idx]
        run_dir: Path = ep["run_dir"]
        records: List[Dict] = ep["records"]

        depths: List[np.ndarray] = []
        goals: List[np.ndarray] = []
        actions: List[np.ndarray] = []

        for rec in records:
            sample_path = resolve_sample_path(run_dir, rec["sample_npz"])
            with np.load(str(sample_path)) as npz:
                depth_m = np.asarray(npz["depth"], dtype=np.float32)
                goal = np.asarray(npz["goal"], dtype=np.float32).reshape(2)
                action = np.asarray(npz["action"], dtype=np.int64).reshape(1)
            action_id = int(action[0])
            manifest_action_id = int(rec.get("action_id", action_id))
            if action_id != manifest_action_id:
                raise ValueError(
                    f"Action mismatch in {sample_path}: npz action={action_id}, "
                    f"manifest action_id={manifest_action_id}"
                )
            if not 0 <= action_id < len(ACTION_NAMES):
                raise ValueError(f"Invalid action id {action_id} in {sample_path}")

            if depth_m.ndim == 3 and depth_m.shape[-1] == 1:
                depth_m = depth_m[..., 0]
            if depth_m.ndim != 2:
                raise ValueError(f"Expected depth [H,W] or [H,W,1], got {depth_m.shape} in {sample_path}")

            if depth_m.shape != (self.resolution, self.resolution):
                # Keep dependency optional; the collector should normally already save this size.
                try:
                    import cv2
                except Exception as exc:
                    raise RuntimeError(
                        f"Depth shape {depth_m.shape} != {(self.resolution, self.resolution)} and cv2 is not available."
                    ) from exc
                depth_m = cv2.resize(
                    depth_m,
                    (self.resolution, self.resolution),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.float32)

            # The collector saves meters. Policy expects [H,W,1] normalized to [0,1].
            depth_norm = np.nan_to_num(
                depth_m,
                nan=self.max_depth_m,
                posinf=self.max_depth_m,
                neginf=0.0,
            )
            depth_norm = np.clip(depth_norm, 0.0, self.max_depth_m) / max(self.max_depth_m, 1e-6)
            depth_norm = depth_norm[..., None].astype(np.float32)

            depths.append(depth_norm)
            goals.append(goal.astype(np.float32))
            actions.append(action.astype(np.int64))

        return {
            "depth": np.stack(depths, axis=0),      # [T,H,W,1]
            "goal": np.stack(goals, axis=0),        # [T,2]
            "actions": np.stack(actions, axis=0),   # [T,1]
            "run_dir": str(run_dir),
        }

    def action_counts(self) -> np.ndarray:
        counts = np.zeros(4, dtype=np.int64)
        for ep in self.episodes:
            for rec in ep["records"]:
                a = int(rec.get("action_id", -1))
                if 0 <= a < 4:
                    counts[a] += 1
        return counts

    def num_steps(self) -> int:
        return int(sum(len(ep["records"]) for ep in self.episodes))


class PadEpisodeCollate:
    def __init__(self, resolution: int):
        self.resolution = int(resolution)

    def __call__(self, batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        bsz = len(batch)
        max_t = max(item["depth"].shape[0] for item in batch)
        H = W = self.resolution

        depth = torch.zeros(bsz, max_t, H, W, 1, dtype=torch.float32)
        goal = torch.zeros(bsz, max_t, 2, dtype=torch.float32)
        actions = torch.zeros(bsz, max_t, 1, dtype=torch.long)
        valid = torch.zeros(bsz, max_t, 1, dtype=torch.bool)
        masks = torch.zeros(bsz, max_t, 1, dtype=torch.bool)
        lengths = torch.zeros(bsz, dtype=torch.long)

        run_dirs = []
        for i, item in enumerate(batch):
            T = item["depth"].shape[0]
            lengths[i] = T
            depth[i, :T] = torch.from_numpy(item["depth"])
            goal[i, :T] = torch.from_numpy(item["goal"])
            actions[i, :T] = torch.from_numpy(item["actions"])
            valid[i, :T] = True
            if T > 1:
                masks[i, 1:T] = True
            run_dirs.append(item["run_dir"])

        return {
            "depth": depth,
            "goal": goal,
            "actions": actions,
            "valid": valid,
            "masks": masks,
            "lengths": lengths,
            "run_dirs": run_dirs,
        }


# -----------------------------------------------------------------------------
# Training utilities.
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(policy) -> Tuple[int, int]:
    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    return total, trainable


def set_freeze_mode(policy, freeze_mode: str, extra_trainable_patterns: Optional[Sequence[str]] = None) -> List[str]:
    for p in policy.parameters():
        p.requires_grad = False

    patterns: List[str]
    if freeze_mode == "action_head":
        patterns = ["action_distribution"]
    elif freeze_mode == "action_head_lstm":
        patterns = ["action_distribution", "net.state_encoder"]
    elif freeze_mode == "action_head_lstm_visual_fc":
        patterns = ["action_distribution", "net.state_encoder", "net.visual_fc"]
    elif freeze_mode == "last_resnet_block":
        patterns = [
            "action_distribution",
            "net.state_encoder",
            "net.visual_fc",
            "net.visual_encoder.backbone.layer4",
            "net.visual_encoder.layer4",
        ]
    elif freeze_mode == "all":
        patterns = [""]
    else:
        raise ValueError(f"Unknown freeze_mode: {freeze_mode}")

    if extra_trainable_patterns:
        patterns.extend([p for p in extra_trainable_patterns if p])

    matched = []
    for name, p in policy.named_parameters():
        if any(pattern == "" or pattern in name for pattern in patterns):
            p.requires_grad = True
            matched.append(name)

    return matched


def maybe_set_frozen_backbone_eval(policy, freeze_mode: str) -> None:
    """Keep frozen perception modules in eval mode to avoid changing norm statistics."""
    if freeze_mode in {"action_head", "action_head_lstm", "action_head_lstm_visual_fc"}:
        for module_path in ["net.visual_encoder"]:
            mod = get_module_by_path(policy, module_path)
            if mod is not None:
                mod.eval()


def get_module_by_path(root, path: str):
    obj = root
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def distribution_log_probs(dist, actions: torch.Tensor) -> torch.Tensor:
    if hasattr(dist, "log_probs"):
        return dist.log_probs(actions)
    if hasattr(dist, "log_prob"):
        return dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
    raise RuntimeError("Action distribution has neither log_probs nor log_prob.")


def distribution_entropy(dist) -> torch.Tensor:
    if hasattr(dist, "entropy"):
        ent = dist.entropy()
        if ent.ndim == 1:
            ent = ent.unsqueeze(-1)
        return ent
    raise RuntimeError("Action distribution has no entropy().")


def distribution_mode(dist) -> torch.Tensor:
    if hasattr(dist, "mode"):
        return dist.mode()
    probs = distribution_probs(dist)
    return probs.argmax(dim=-1, keepdim=True)


def distribution_probs(dist) -> torch.Tensor:
    if hasattr(dist, "probs"):
        return dist.probs
    if hasattr(dist, "distribution") and hasattr(dist.distribution, "probs"):
        return dist.distribution.probs
    if hasattr(dist, "logits"):
        return torch.softmax(dist.logits, dim=-1)
    if hasattr(dist, "distribution") and hasattr(dist.distribution, "logits"):
        return torch.softmax(dist.distribution.logits, dim=-1)
    raise RuntimeError("Cannot extract probs/logits from action distribution.")


def categorical_kl_old_new(old_dist, new_dist, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample KL(old || new), shape [B]."""
    old_p = distribution_probs(old_dist).detach().clamp_min(eps)
    new_p = distribution_probs(new_dist).clamp_min(eps)
    return (old_p * (old_p.log() - new_p.log())).sum(dim=-1)


def build_initial_hidden(policy, batch_size: int, hidden_size: int, device: torch.device) -> torch.Tensor:
    # Matches falcon_ros_bridge.py: [B, num_recurrent_layers, hidden_size]
    num_layers = int(policy.net.num_recurrent_layers)
    return torch.zeros(batch_size, num_layers, hidden_size, device=device)


def forward_sequence_loss(
    policy,
    batch: Dict[str, torch.Tensor],
    args,
    device: torch.device,
    old_policy=None,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    depth = batch["depth"].to(device, non_blocking=True)
    goal = batch["goal"].to(device, non_blocking=True)
    actions = batch["actions"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    masks = batch["masks"].to(device, non_blocking=True)

    B, T = depth.shape[:2]
    hidden = build_initial_hidden(policy, B, args.hidden_size, device)
    old_hidden = build_initial_hidden(old_policy, B, args.hidden_size, device) if old_policy is not None else None

    prev_actions = torch.zeros(B, 1, dtype=torch.long, device=device)

    bc_numer = torch.zeros((), dtype=torch.float32, device=device)
    kl_numer = torch.zeros((), dtype=torch.float32, device=device)
    ent_numer = torch.zeros((), dtype=torch.float32, device=device)
    bc_denom = torch.zeros((), dtype=torch.float32, device=device)
    denom = torch.zeros((), dtype=torch.float32, device=device)

    total_correct = 0.0
    total_count = 0.0
    per_action_correct = torch.zeros(4, dtype=torch.float64, device=device)
    per_action_count = torch.zeros(4, dtype=torch.float64, device=device)

    for t in range(T):
        valid_t = valid[:, t, 0]
        if not bool(valid_t.any().item()):
            break

        obs_t = {
            args.depth_obs_key: depth[:, t],
            args.goal_obs_key: goal[:, t],
        }
        masks_t = masks[:, t]
        expert_t = actions[:, t]

        features, hidden, _ = policy.net(obs_t, hidden, prev_actions, masks_t)
        dist = policy.action_distribution(features)
        logp = distribution_log_probs(dist, expert_t).squeeze(-1)
        neglogp = -logp

        weights_t = torch.ones_like(neglogp)
        if class_weights is not None:
            weights_t = class_weights[expert_t.squeeze(-1)]

        valid_float = valid_t.float()
        bc_numer = bc_numer + (neglogp * weights_t * valid_float).sum()
        bc_denom = bc_denom + (weights_t * valid_float).sum()
        denom = denom + valid_float.sum()

        if args.entropy_coef != 0.0:
            ent = distribution_entropy(dist).squeeze(-1)
            ent_numer = ent_numer + (ent * valid_float).sum()

        if old_policy is not None and args.kl_coef > 0.0:
            with torch.no_grad():
                old_features, old_hidden, _ = old_policy.net(obs_t, old_hidden, prev_actions, masks_t)
                old_dist = old_policy.action_distribution(old_features)
            kl = categorical_kl_old_new(old_dist, dist)
            kl_numer = kl_numer + (kl * valid_float).sum()

        with torch.no_grad():
            pred = distribution_mode(dist)
            correct = (pred.squeeze(-1) == expert_t.squeeze(-1)) & valid_t
            total_correct += float(correct.sum().item())
            total_count += float(valid_t.sum().item())
            for a in range(4):
                mask_a = (expert_t.squeeze(-1) == a) & valid_t
                per_action_count[a] += mask_a.sum().double()
                per_action_correct[a] += (correct & mask_a).sum().double()

        # Teacher forcing: next previous action is expert action for valid episodes.
        prev_actions = torch.where(valid_t.view(B, 1), expert_t.detach(), torch.zeros_like(prev_actions))

    bc_denom = bc_denom.clamp_min(1.0)
    denom = denom.clamp_min(1.0)
    bc_loss = bc_numer / bc_denom
    kl_loss = kl_numer / denom
    entropy = ent_numer / denom
    loss = bc_loss + args.kl_coef * kl_loss - args.entropy_coef * entropy

    per_action_acc = []
    for a in range(4):
        c = float(per_action_count[a].item())
        if c <= 0:
            per_action_acc.append(float("nan"))
        else:
            per_action_acc.append(float((per_action_correct[a] / per_action_count[a]).item()))

    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "bc_loss": float(bc_loss.detach().cpu().item()),
        "kl_loss": float(kl_loss.detach().cpu().item()),
        "entropy": float(entropy.detach().cpu().item()),
        "acc": float(total_correct / max(total_count, 1.0)),
        "count": float(total_count),
        "acc_stop": per_action_acc[0],
        "acc_forward": per_action_acc[1],
        "acc_left": per_action_acc[2],
        "acc_right": per_action_acc[3],
        "correct_stop": float(per_action_correct[0].item()),
        "correct_forward": float(per_action_correct[1].item()),
        "correct_left": float(per_action_correct[2].item()),
        "correct_right": float(per_action_correct[3].item()),
        "count_stop": float(per_action_count[0].item()),
        "count_forward": float(per_action_count[1].item()),
        "count_left": float(per_action_count[2].item()),
        "count_right": float(per_action_count[3].item()),
    }
    return loss, metrics


def merge_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    if len(metric_list) == 0:
        return {}
    total_count = sum(m.get("count", 0.0) for m in metric_list)
    out: Dict[str, float] = {}
    acc_count_keys = {
        "acc_stop": ("correct_stop", "count_stop"),
        "acc_forward": ("correct_forward", "count_forward"),
        "acc_left": ("correct_left", "count_left"),
        "acc_right": ("correct_right", "count_right"),
    }
    for key in metric_list[0].keys():
        if key == "count":
            out[key] = total_count
            continue
        if key.startswith("correct_") or key.startswith("count_"):
            out[key] = float(sum(m.get(key, 0.0) for m in metric_list))
            continue
        if key in acc_count_keys:
            correct_key, count_key = acc_count_keys[key]
            correct = sum(m.get(correct_key, 0.0) for m in metric_list)
            count = sum(m.get(count_key, 0.0) for m in metric_list)
            out[key] = float("nan") if count <= 0 else float(correct / count)
            continue
        if key.startswith("acc"):
            vals = [m[key] for m in metric_list if not math.isnan(m.get(key, float("nan")))]
            if len(vals) == 0:
                out[key] = float("nan")
            else:
                out[key] = float(sum(vals) / len(vals))
        else:
            if total_count > 0:
                out[key] = float(
                    sum(m[key] * m.get("count", 0.0) for m in metric_list) / total_count
                )
            else:
                out[key] = float(np.mean([m[key] for m in metric_list]))
    return out


def run_one_epoch(
    policy,
    loader: DataLoader,
    args,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    old_policy=None,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    train = optimizer is not None
    if train:
        policy.train()
        maybe_set_frozen_backbone_eval(policy, args.freeze_mode)
    else:
        policy.eval()

    metrics_all: List[Dict[str, float]] = []
    for batch_idx, batch in enumerate(loader):
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = forward_sequence_loss(policy, batch, args, device, old_policy, class_weights)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad],
                    max_norm=args.max_grad_norm,
                )
            optimizer.step()
        else:
            with torch.no_grad():
                loss, metrics = forward_sequence_loss(policy, batch, args, device, old_policy, class_weights)
        metrics_all.append(metrics)

        if train and args.log_interval > 0 and (batch_idx + 1) % args.log_interval == 0:
            print(
                f"  batch {batch_idx + 1:04d}/{len(loader):04d} "
                f"loss={metrics['loss']:.4f} bc={metrics['bc_loss']:.4f} "
                f"kl={metrics['kl_loss']:.4f} acc={metrics['acc']:.3f}"
            )

    return merge_metrics(metrics_all)


def split_run_dirs(run_dirs: List[Path], val_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    run_dirs = list(run_dirs)
    rng = random.Random(seed)
    rng.shuffle(run_dirs)
    if val_ratio <= 0 or len(run_dirs) <= 1:
        return run_dirs, []
    n_val = max(1, int(round(len(run_dirs) * val_ratio)))
    n_val = min(n_val, len(run_dirs) - 1)
    return run_dirs[n_val:], run_dirs[:n_val]


def make_class_weights(counts: np.ndarray, device: torch.device) -> torch.Tensor:
    counts_f = counts.astype(np.float64)
    counts_f = np.maximum(counts_f, 1.0)
    inv = counts_f.sum() / counts_f
    inv = inv / inv.mean()
    return torch.tensor(inv, dtype=torch.float32, device=device)


def save_checkpoint(
    output_checkpoint: str,
    policy,
    args,
    epoch: int,
    best_val_loss: float,
    history: List[Dict],
    train_run_dirs: Sequence[Path],
    val_run_dirs: Sequence[Path],
) -> None:
    output_path = Path(output_checkpoint).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = {
        "actor_critic." + k: v.detach().cpu()
        for k, v in policy.state_dict().items()
    }
    ckpt = {
        "state_dict": state_dict,
        "bc_finetune_meta": {
            "epoch": int(epoch),
            "best_val_loss": float(best_val_loss),
            "args": vars(args),
            "train_run_dirs": [str(p) for p in train_run_dirs],
            "val_run_dirs": [str(p) for p in val_run_dirs],
            "action_names": ACTION_NAMES,
            "note": "Saved with actor_critic. prefix for compatibility with falcon_ros_bridge.py loader.",
        },
    }
    torch.save(ckpt, str(output_path))

    history_path = output_path.with_suffix(output_path.suffix + ".history.json")
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def format_metrics(prefix: str, m: Dict[str, float]) -> str:
    if not m:
        return f"{prefix}: N/A"
    parts = [
        f"{prefix}: loss={m['loss']:.4f}",
        f"bc={m['bc_loss']:.4f}",
        f"kl={m['kl_loss']:.4f}",
        f"acc={m['acc']:.3f}",
    ]
    for k, label in [
        ("acc_stop", "stop"),
        ("acc_forward", "forward"),
        ("acc_left", "left"),
        ("acc_right", "right"),
    ]:
        v = m.get(k, float("nan"))
        if not math.isnan(v):
            parts.append(f"{label}={v:.3f}")
    return " | ".join(parts)


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Offline BC fine-tuning for Falcon PointNavResNetPolicy")

    # Paths.
    p.add_argument("--checkpoint", type=str, required=True, help="Original Falcon/DD-PPO checkpoint.")
    p.add_argument(
        "--data_root",
        type=str,
        nargs="+",
        required=True,
        help="One or more trajectory roots/run_dirs. The script searches for manifest.jsonl.",
    )
    p.add_argument("--output_checkpoint", type=str, required=True)
    p.add_argument("--habitat_baselines_path", type=str, default="/home/mobile/ranger_nav/habitat-baselines/")
    p.add_argument("--habitat_lab_path", type=str, default="/home/mobile/ranger_nav/habitat-lab/")
    p.add_argument("--project_path", type=str, default="/home/mobile/ranger_nav")

    # Policy construction. Defaults match falcon_ros_bridge.py.
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max_depth_m", type=float, default=10.0)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_recurrent_layers", type=int, default=2)
    p.add_argument("--backbone", type=str, default="resnet50")
    p.add_argument("--rnn_type", type=str, default="LSTM")
    p.add_argument("--depth_obs_key", type=str, default="articulated_agent_jaw_depth")
    p.add_argument("--goal_obs_key", type=str, default="pointgoal_with_gps_compass")
    p.add_argument("--strict_checkpoint", action="store_true")

    # Dataset options.
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--overfit_n_episodes", type=int, default=0, help="Use only N episodes and disable val split if >0.")
    p.add_argument("--max_consecutive_stop", type=int, default=3, help="Trim stop label runs beyond this count. -1 disables.")
    p.add_argument("--min_episode_len", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)

    # Training options.
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4, help="Number of trajectories per batch.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument(
        "--freeze_mode",
        type=str,
        default="action_head",
        choices=["action_head", "action_head_lstm", "action_head_lstm_visual_fc", "last_resnet_block", "all"],
    )
    p.add_argument(
        "--extra_trainable_pattern",
        type=str,
        action="append",
        default=[],
        help="Additional substring pattern for parameters to unfreeze. Can be passed multiple times.",
    )
    p.add_argument("--kl_coef", type=float, default=0.02, help="KL(old_policy || new_policy) coefficient.")
    p.add_argument("--entropy_coef", type=float, default=0.0)
    p.add_argument("--class_balance", action="store_true", help="Use inverse-frequency action weights for BC loss.")
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_every_epoch", action="store_true")
    p.add_argument("--device", type=str, default="cuda:0")

    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print("[INFO] device:", device)

    run_dirs = discover_run_dirs(args.data_root)
    if args.overfit_n_episodes > 0:
        run_dirs = run_dirs[: args.overfit_n_episodes]
        train_run_dirs, val_run_dirs = run_dirs, []
        print(f"[DATA] overfit mode: using {len(train_run_dirs)} episodes, no validation split")
    else:
        train_run_dirs, val_run_dirs = split_run_dirs(run_dirs, args.val_ratio, args.seed)

    print(f"[DATA] discovered episodes: {len(run_dirs)}")
    print(f"[DATA] train episodes: {len(train_run_dirs)} | val episodes: {len(val_run_dirs)}")

    train_ds = FalconImitationEpisodeDataset(
        train_run_dirs,
        resolution=args.resolution,
        max_depth_m=args.max_depth_m,
        max_consecutive_stop=args.max_consecutive_stop,
        min_episode_len=args.min_episode_len,
    )
    val_ds = None
    if len(val_run_dirs) > 0:
        val_ds = FalconImitationEpisodeDataset(
            val_run_dirs,
            resolution=args.resolution,
            max_depth_m=args.max_depth_m,
            max_consecutive_stop=args.max_consecutive_stop,
            min_episode_len=args.min_episode_len,
        )

    print(f"[DATA] train steps: {train_ds.num_steps()} | action counts: {train_ds.action_counts().tolist()}")
    if val_ds is not None:
        print(f"[DATA] val steps:   {val_ds.num_steps()} | action counts: {val_ds.action_counts().tolist()}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=PadEpisodeCollate(args.resolution),
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=PadEpisodeCollate(args.resolution),
        )

    policy = build_policy(args, device)
    policy.train()

    old_policy = None
    if args.kl_coef > 0.0:
        old_policy = copy.deepcopy(policy).to(device)
        old_policy.eval()
        for p in old_policy.parameters():
            p.requires_grad = False
        print(f"[KL] enabled with coef={args.kl_coef}")

    matched = set_freeze_mode(policy, args.freeze_mode, args.extra_trainable_pattern)
    total_params, trainable_params = count_parameters(policy)
    print(f"[FREEZE] mode={args.freeze_mode}")
    print(f"[FREEZE] trainable parameter tensors: {len(matched)}")
    print(f"[FREEZE] trainable params: {trainable_params:,} / {total_params:,} ({100.0 * trainable_params / max(total_params, 1):.2f}%)")
    print("[FREEZE] first trainable names:")
    for name in list(matched)[:30]:
        print("  ", name)
    if len(matched) == 0:
        raise RuntimeError("No parameters selected for training. Check --freeze_mode / --extra_trainable_pattern.")

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    class_weights = None
    if args.class_balance:
        class_weights = make_class_weights(train_ds.action_counts(), device)
        print("[LOSS] class weights:", class_weights.detach().cpu().numpy().round(4).tolist())

    history: List[Dict] = []
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(
            policy,
            train_loader,
            args,
            device,
            optimizer=optimizer,
            old_policy=old_policy,
            class_weights=class_weights,
        )
        val_metrics = {}
        if val_loader is not None:
            val_metrics = run_one_epoch(
                policy,
                val_loader,
                args,
                device,
                optimizer=None,
                old_policy=old_policy,
                class_weights=class_weights,
            )

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        print(f"[EPOCH {epoch:03d}] {format_metrics('train', train_metrics)}")
        if val_loader is not None:
            print(f"[EPOCH {epoch:03d}] {format_metrics('val', val_metrics)}")

        monitor_loss = val_metrics.get("loss", train_metrics["loss"])
        improved = monitor_loss < best_val_loss
        if improved:
            best_val_loss = monitor_loss
            best_epoch = epoch
            save_checkpoint(
                args.output_checkpoint,
                policy,
                args,
                epoch,
                best_val_loss,
                history,
                train_run_dirs,
                val_run_dirs,
            )
            print(f"[SAVE] best checkpoint updated at epoch {epoch}: {args.output_checkpoint}")

        if args.save_every_epoch:
            stem = Path(args.output_checkpoint).expanduser().resolve()
            epoch_path = str(stem.with_suffix(stem.suffix + f".epoch{epoch:03d}"))
            save_checkpoint(
                epoch_path,
                policy,
                args,
                epoch,
                best_val_loss,
                history,
                train_run_dirs,
                val_run_dirs,
            )

    print(f"[DONE] best_epoch={best_epoch} best_loss={best_val_loss:.6f}")
    print(f"[DONE] best checkpoint: {args.output_checkpoint}")


if __name__ == "__main__":
    main()
