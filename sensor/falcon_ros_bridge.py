#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Falcon ROS bridge.

Data flow (runtime):
1) Subscribe Depth + goal signal (polar topic PointStamped).
2) Build policy observation in Falcon/PointNav format.
3) Run policy inference (Discrete(4)).
4) Map discrete action to Twist and publish to cmd_vel.

Safety behavior:
- If goal input is missing/too old, publish stop.
- If callback fails, publish stop.
- If input stream stalls (watchdog timeout), publish stop.
"""

import argparse
from typing import Dict, Optional, Tuple
from collections import deque

import cv2
import numpy as np
import rospy
import torch
from geometry_msgs.msg import PointStamped, Twist
from gym.spaces import Box
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete
from sensor_msgs.msg import Image
from std_msgs.msg import Header

import sys
sys.path.append("/home/mobile/ranger_nav/habitat-baselines/")
sys.path.append("/home/mobile/ranger_nav/habitat-lab/")
sys.path.append("/home/mobile/ranger_nav")

from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
from habitat_baselines.utils.common import batch_obs


def _extract_actor_critic_state_dict(ckpt_obj: Dict) -> Dict[str, torch.Tensor]:
    """
    Accept different checkpoint layouts and return state_dict keys
    compatible with PointNavResNetPolicy.
    """
    src = ckpt_obj.get("state_dict", ckpt_obj)
    if not isinstance(src, dict):
        raise RuntimeError("Unsupported checkpoint format: state_dict is not a dict.")

    out = {}
    for k, v in src.items():
        if not isinstance(k, str):
            continue
        if "actor_critic." in k:
            kk = k.split("actor_critic.", 1)[1]
            out[kk] = v
        elif k.startswith("net.") or k.startswith("action_distribution.") or k.startswith("critic."):
            out[k] = v
    if len(out) == 0:
        # Fallback: assume keys are already policy keys.
        out = src
    return out


class FalconRosBridge(object):
    """Bridge ROS sensor streams to Falcon policy and publish cmd_vel."""

    def __init__(self, args):
        rospy.init_node("falcon_ros_bridge", anonymous=False)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rospy.loginfo("Falcon device: %s", str(self.device))

        self.resolution = args.resolution
        self.max_depth_m = args.max_depth_m
        self.deterministic = args.deterministic
        self.require_strict_ckpt = args.strict_checkpoint
        self.debug_mapping = args.debug_mapping
        self.debug_depth = args.debug_depth
        self.theta_deadband_rad = args.theta_deadband_rad
        self.theta_excess_offset_rad = args.theta_excess_offset_rad

        # Policy obs keys should match your social_nav_v2 config.
        self.depth_key = args.depth_obs_key
        self.goal_key = args.goal_obs_key

        self.latest_polar_msg = None
        self.polar_buffer = deque(maxlen=max(10, args.polar_buffer_size))
        self.last_obs_time = rospy.Time(0)
        self.data_timeout_sec = args.data_timeout_sec
        self.max_polar_age_sec = args.max_polar_age_sec

        self.actor_critic = self._build_policy(
            checkpoint_path=args.checkpoint,
            hidden_size=args.hidden_size,
            num_recurrent_layers=args.num_recurrent_layers,
            backbone=args.backbone,
            rnn_type=args.rnn_type,
        )
        self.actor_critic.eval()

        self.hidden_states = torch.zeros(
            1,
            self.actor_critic.net.num_recurrent_layers,
            args.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)

        self.cmd_pub = rospy.Publisher(args.cmd_vel_topic, Twist, queue_size=10)
        self.debug_obs_pub = rospy.Publisher(args.debug_obs_topic, Header, queue_size=10)

        self.polar_sub = rospy.Subscriber(
            args.polar_topic, PointStamped, self._polar_cb, queue_size=20
        )
        self.depth_sub = rospy.Subscriber(args.depth_topic, Image, self._cb_depth, queue_size=10)

        self.action_to_cmd = {
            0: (0.0, 0.0),  # stop
            1: (args.forward_speed, 0.0),  # forward
            2: (0.0, args.turn_speed),  # turn left
            3: (0.0, -args.turn_speed),  # turn right
        }

        rospy.loginfo("Falcon ROS bridge started.")
        rospy.loginfo("Subscribe: %s, %s", args.depth_topic, args.polar_topic)
        rospy.loginfo("Publish:   %s", args.cmd_vel_topic)
        self.watchdog = rospy.Timer(rospy.Duration(0.05), self._watchdog_cb)

    def _build_policy(
        self,
        checkpoint_path: str,
        hidden_size: int,
        num_recurrent_layers: int,
        backbone: str,
        rnn_type: str,
    ):
        # Build observation/action spaces that match the training setup.
        spaces = {
            self.goal_key: Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            ),
            self.depth_key: Box(
                low=0.0,
                high=1.0,
                shape=(self.resolution, self.resolution, 1),
                dtype=np.float32,
            ),
        }

        observation_space = SpaceDict(spaces)
        action_space = Discrete(4)

        policy = PointNavResNetPolicy(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            rnn_type=rnn_type,
            backbone=backbone,
            normalize_visual_inputs=False,
        ).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device,weights_only=False)
        ckpt = ckpt[0]["state_dict"]
        policy_sd = _extract_actor_critic_state_dict(ckpt)
        missing, unexpected = policy.load_state_dict(policy_sd, strict=False)

        rospy.logwarn("Checkpoint loaded with strict=False.")
        rospy.logwarn("Missing keys: %d, Unexpected keys: %d", len(missing), len(unexpected))
        if self.require_strict_ckpt and (len(missing) > 0 or len(unexpected) > 0):
            raise RuntimeError(
                "Checkpoint key mismatch: missing={} unexpected={}".format(
                    len(missing), len(unexpected)
                )
            )
        return policy

    @staticmethod
    def _ros_image_to_numpy(msg: Image) -> np.ndarray:
        # Convert common ROS Image encodings to numpy without cv_bridge.
        if msg.encoding == "16UC1":
            row_bytes = msg.width * 2
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
            arr = raw[:, :row_bytes].copy().view(np.uint16).reshape(msg.height, msg.width)
        elif msg.encoding == "32FC1":
            row_bytes = msg.width * 4
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
            arr = raw[:, :row_bytes].copy().view(np.float32).reshape(msg.height, msg.width)
        elif msg.encoding in ("rgb8", "bgr8"):
            row_bytes = msg.width * 3
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
            arr = raw[:, :row_bytes].copy().reshape(msg.height, msg.width, 3)
        else:
            raise ValueError("Unsupported image encoding: {}".format(msg.encoding))

        if msg.is_bigendian:
            arr = arr.byteswap().newbyteorder()
        return arr

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

    def _depth_msg_to_norm_depth(
        self, depth_msg: Image
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        # Accept either millimeter uint16 depth or meter float32 depth, then
        # normalize to [0, 1] and resize to policy resolution.
        debug = {
            "encoding": depth_msg.encoding,
            "raw_shape": None,
            "raw_dtype": None,
            "raw_unit": "m",
            "raw_stats": None,
            "depth_m_stats": None,
            "norm_shape": None,
            "norm_dtype": None,
            "norm_stats": None,
        }

        if depth_msg.encoding == "16UC1":
            depth_u16 = self._ros_image_to_numpy(depth_msg)
            debug["raw_shape"] = tuple(depth_u16.shape)
            debug["raw_dtype"] = str(depth_u16.dtype)
            debug["raw_unit"] = "mm"
            debug["raw_stats"] = self._depth_stats(depth_u16)
            depth_m = depth_u16.astype(np.float32) * 0.001
        else:
            depth_f32 = self._ros_image_to_numpy(depth_msg)
            debug["raw_shape"] = tuple(depth_f32.shape)
            debug["raw_dtype"] = str(depth_f32.dtype)
            debug["raw_unit"] = "m"
            debug["raw_stats"] = self._depth_stats(depth_f32)
            depth_m = depth_f32.astype(np.float32)

        depth_m = np.nan_to_num(depth_m, nan=self.max_depth_m, posinf=self.max_depth_m, neginf=0.0)
        depth_m = np.clip(depth_m, 0.0, self.max_depth_m)
        debug["depth_m_stats"] = self._depth_stats(depth_m)
        depth_norm = depth_m / self.max_depth_m
        depth_norm = cv2.resize(depth_norm, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
        depth_norm = np.expand_dims(depth_norm.astype(np.float32), axis=-1)
        debug["norm_shape"] = tuple(depth_norm.shape)
        debug["norm_dtype"] = str(depth_norm.dtype)
        debug["norm_stats"] = self._depth_stats(depth_norm)
        return depth_norm, debug

    def _build_obs(
        self, depth_msg: Image, polar_msg: PointStamped
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
        # Polar convention from sensor/polar_distance.py: x=r, y=theta.
        r = np.float32(polar_msg.point.x)
        theta = np.float32(self._shape_theta(float(polar_msg.point.y)))
        depth_norm, depth_debug = self._depth_msg_to_norm_depth(depth_msg)

        obs = {
            self.depth_key: depth_norm,
            self.goal_key: np.array([r, theta], dtype=np.float32),
        }
        return obs, depth_debug

    def _shape_theta(self, theta: float) -> float:
        # Piecewise angular shaping for discrete control:
        # - if |theta| <= deadband: treat as 0
        # - else: keep sign and subtract a fixed offset from magnitude
        a = abs(theta)
        if a <= self.theta_deadband_rad:
            return 0.0
        reduced = a - self.theta_excess_offset_rad
        if reduced < 0.0:
            reduced = 0.0
        return reduced if theta >= 0.0 else -reduced

    def _infer_action(
        self, obs: Dict[str, np.ndarray]
    ) -> Tuple[int, Optional[np.ndarray]]:
        # Recurrent policy inference:
        # hidden_states/prev_actions/not_done_masks are carried across timesteps.
        batch = batch_obs([obs], device=self.device)
        with torch.no_grad():
            features, next_hidden_states, _ = self.actor_critic.net(
                batch,
                self.hidden_states,
                self.prev_actions,
                self.not_done_masks,
            )
            distribution = self.actor_critic.action_distribution(features)
            if self.deterministic:
                if self.actor_critic.action_distribution_type == "categorical":
                    actions = distribution.mode()
                elif self.actor_critic.action_distribution_type == "gaussian":
                    actions = distribution.mean
                else:
                    actions = distribution.sample()
            else:
                actions = distribution.sample()

            self.hidden_states = next_hidden_states
            self.not_done_masks.fill_(True)
            self.prev_actions.copy_(actions)

            probs = None
            if self.actor_critic.action_distribution_type == "categorical":
                probs = distribution.probs[0].detach().cpu().numpy()

        return int(actions[0][0].item()), probs
    # 放到 FalconRosBridge 类里，例如 _infer_action 后面
    @staticmethod
    def _fmt_stats(stats: Dict[str, float]) -> str:
        return (
            "valid={:.1f}% min={:.3f} max={:.3f} mean={:.3f} p50={:.3f} p95={:.3f} zero={:.1f}%".format(
                100.0 * stats["valid_ratio"],
                stats["min"],
                stats["max"],
                stats["mean"],
                stats["p50"],
                stats["p95"],
                100.0 * stats["zero_ratio"],
            )
        )

    def _fmt_action_probs(self, probs: Optional[np.ndarray]) -> str:
        if probs is None:
            return "N/A(non-categorical)"
        labels = ["stop", "forward", "left", "right"]
        items = []
        for i, p in enumerate(probs.tolist()):
            name = labels[i] if i < len(labels) else "a{}".format(i)
            items.append("{}:{:.3f}".format(name, float(p)))
        return " ".join(items)

    def _debug_print_once(
        self,
        obs: Dict[str, np.ndarray],
        act_id: int,
        theta_raw: float,
        probs: Optional[np.ndarray],
        depth_debug: Dict[str, object],
    ):
        g = obs[self.goal_key]
        d = obs[self.depth_key]
        lin, ang = self.action_to_cmd.get(act_id, (0.0, 0.0))

        if self.debug_depth:
            rospy.loginfo(
                "[DBG_DEPTH] ros(enc={}, raw={} {}, unit={}) raw_stats=[{}] depth_m_stats=[{}] "
                "falcon_expected(shape=({}, {}, 1), dtype=float32, norm=[0,1]) falcon_input(actual={} {}) norm_stats=[{}]".format(
                    depth_debug["encoding"],
                    depth_debug["raw_shape"],
                    depth_debug["raw_dtype"],
                    depth_debug["raw_unit"],
                    self._fmt_stats(depth_debug["raw_stats"]),
                    self._fmt_stats(depth_debug["depth_m_stats"]),
                    self.resolution,
                    self.resolution,
                    depth_debug["norm_shape"],
                    depth_debug["norm_dtype"],
                    self._fmt_stats(depth_debug["norm_stats"]),
                )
            )

        rospy.loginfo(
            "[DBG_ACT] goal[r,theta]=[{:.3f}, {:.3f}] theta_raw={:.3f} depth_shape={} depth[min,max]=[{:.3f},{:.3f}] "
            "act_id={} cmd=({:.3f},{:.3f}) probs=[{}]".format(
                float(g[0]), float(g[1]), float(theta_raw),
                tuple(d.shape), float(d.min()), float(d.max()),
                int(act_id), float(lin), float(ang),
                self._fmt_action_probs(probs),
            )
        )


    def _publish_cmd(self, action_id: int):
        # Action id -> (linear x, angular z).
        lin, ang = self.action_to_cmd.get(action_id, (0.0, 0.0))
        tw = Twist()
        tw.linear.x = lin
        tw.angular.z = ang
        self.cmd_pub.publish(tw)

    def _publish_stop(self):
        tw = Twist()
        self.cmd_pub.publish(tw)

    def _polar_cb(self, polar_msg: PointStamped):
        self.latest_polar_msg = polar_msg
        self.polar_buffer.append(polar_msg)

    def _pick_polar_for_stamp(self, target_stamp: rospy.Time):
        # Pick the temporally closest polar message to current image timestamp.
        if len(self.polar_buffer) == 0:
            return None

        # If image has no timestamp, fallback to latest.
        if target_stamp == rospy.Time():
            return self.polar_buffer[-1]

        best = None
        best_dt = None
        for msg in self.polar_buffer:
            if msg.header.stamp == rospy.Time():
                continue
            dt = abs((target_stamp - msg.header.stamp).to_sec())
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best = msg

        if best is None:
            return self.polar_buffer[-1]
        if best_dt is not None and best_dt > self.max_polar_age_sec:
            return None
        return best

    def _emit_heartbeat(self):
        # Lightweight debug pulse indicating inference loop is alive.
        hdr = Header()
        hdr.stamp = rospy.Time.now()
        hdr.frame_id = "falcon_obs_ok"
        self.debug_obs_pub.publish(hdr)

    def _process_one(self, depth_msg: Image, polar_msg: PointStamped = None):
        # Single end-to-end control step: select goal -> build obs -> infer -> publish cmd.
        if polar_msg is None:
            polar_msg = self._pick_polar_for_stamp(depth_msg.header.stamp)
        if polar_msg is None:
            rospy.logwarn_throttle(2.0, "No polar message received yet on polar topic.")
            self._publish_stop()
            return
        try:
            theta_raw = float(polar_msg.point.y)
            obs, depth_debug = self._build_obs(depth_msg=depth_msg, polar_msg=polar_msg)
            act_id, probs = self._infer_action(obs)
            if self.debug_mapping or self.debug_depth:
                self._debug_print_once(obs, act_id, theta_raw, probs, depth_debug)
            self._publish_cmd(act_id)
            self.last_obs_time = rospy.Time.now()
            self._emit_heartbeat()
        except Exception as e:
            self._publish_stop()
            rospy.logerr_throttle(1.0, "Falcon ROS bridge callback failed: %s", str(e))

    def _cb_depth(self, depth_msg: Image):
        self._process_one(depth_msg=depth_msg)

    def _watchdog_cb(self, _event):
        # Fail-safe: stop robot if no successful inference for too long.
        if self.last_obs_time == rospy.Time(0):
            return
        dt = (rospy.Time.now() - self.last_obs_time).to_sec()
        if dt > self.data_timeout_sec:
            self._publish_stop()
            rospy.logwarn_throttle(1.0, "Input timeout %.3fs > %.3fs, publish stop.", dt, self.data_timeout_sec)

    def spin(self):
        rospy.spin()


def parse_args():
    p = argparse.ArgumentParser(description="ROS Depth+Polar -> Falcon -> cmd_vel bridge")
    p.add_argument("--checkpoint", type=str, required=True)
    # Backward-compatibility flags kept for old launch scripts.
    # They are ignored because this bridge is now fixed to depth + polar topic.
    p.add_argument("--input_type", type=str, default="depth", choices=["depth", "rgbd"])
    p.add_argument("--polar_source", type=str, default="topic", choices=["topic", "detections"])

    p.add_argument("--depth_topic", type=str, default="/camera/aligned_depth_to_color/image_raw")
    p.add_argument("--polar_topic", type=str, default="/tag_polar")
    p.add_argument("--cmd_vel_topic", type=str, default="/cmd_vel")
    p.add_argument("--debug_obs_topic", type=str, default="/falcon/obs_heartbeat")

    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max_depth_m", type=float, default=10.0)

    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_recurrent_layers", type=int, default=2)
    p.add_argument("--backbone", type=str, default="resnet50")
    p.add_argument("--rnn_type", type=str, default="LSTM")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--strict_checkpoint", action="store_true")
    p.add_argument("--debug_mapping", action="store_true")
    p.add_argument("--debug_depth", action="store_true")
    p.add_argument("--data_timeout_sec", type=float, default=0.3)
    p.add_argument("--max_polar_age_sec", type=float, default=0.12)
    p.add_argument("--polar_buffer_size", type=int, default=100)

    # Default to non-agent-prefixed keys used by PointNavResNetPolicy sensor handling.
    p.add_argument("--depth_obs_key", type=str, default="articulated_agent_jaw_depth")
    p.add_argument("--goal_obs_key", type=str, default="pointgoal_with_gps_compass")

    p.add_argument("--forward_speed", type=float, default=0.3)
    p.add_argument("--turn_speed", type=float, default=0.3)
    p.add_argument("--theta_deadband_rad", type=float, default=0.15)
    p.add_argument("--theta_excess_offset_rad", type=float, default=0.15)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    node = FalconRosBridge(args)
    node.spin()
