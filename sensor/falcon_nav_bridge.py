#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Falcon bridge for the generic /nav_bridge contract.

Runtime data flow:
1) Subscribe RealSense depth image.
2) Subscribe relative goal PoseStamped in base_link.
3) Convert relative goal (x, y) to Falcon pointgoal [r, theta].
4) Run Falcon policy inference.
5) Publish discrete UInt8 action for the platform-side command mapper.

This bridge intentionally does not publish /cmd_vel by default. The platform
should map the discrete action to its own low-level motion command.
"""

import argparse
from collections import deque
import json
import math
import os
import sys
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import rospy
import torch
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Header, UInt8
from sensor_msgs.msg import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for path in (
    SCRIPT_DIR,
    REPO_ROOT,
    os.path.join(REPO_ROOT, "habitat-baselines"),
    os.path.join(REPO_ROOT, "habitat-lab"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from falcon_ros_bridge import FalconRosBridge  # noqa: E402


STOP = 0
FORWARD = 1
LEFT = 2
RIGHT = 3


class FalconNavBridge(FalconRosBridge):
    """Bridge nav_bridge relative goals + RealSense depth to discrete actions."""

    def __init__(self, args):
        rospy.init_node("falcon_nav_bridge", anonymous=False)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rospy.loginfo("Falcon device: %s", str(self.device))

        self.resolution = args.resolution
        self.max_depth_m = args.max_depth_m
        self.depth_32fc1_unit = args.depth_32fc1_unit
        self.deterministic = args.deterministic
        self.require_strict_ckpt = args.strict_checkpoint
        self.debug_mapping = args.debug_mapping
        self.debug_depth = args.debug_depth
        self.debug_depth_dump_dir = args.debug_depth_dump_dir
        self._depth_sample_saved = False
        self.replay_dump_enabled = args.replay_dump_enabled
        self.replay_dump_dir = args.replay_dump_dir
        self.replay_dump_limit = args.replay_dump_limit
        self._replay_dump_count = 0
        self._replay_dump_limit_reached = False

        self.depth_key = args.depth_obs_key
        self.goal_key = args.goal_obs_key
        self.goal_distance_mode = args.goal_distance_mode
        self.goal_buffer = deque(maxlen=max(10, args.goal_buffer_size))
        self.latest_goal_msg = None
        self.last_obs_time = rospy.Time(0)
        self.data_timeout_sec = args.data_timeout_sec
        self.max_goal_age_sec = args.max_goal_age_sec
        self.ignore_goal_valid = args.ignore_goal_valid
        self.goal_valid = bool(args.ignore_goal_valid)

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

        self.command_pub = rospy.Publisher(args.command_topic, UInt8, queue_size=10)
        self.debug_obs_pub = rospy.Publisher(args.debug_obs_topic, Header, queue_size=10)

        self.goal_sub = rospy.Subscriber(
            args.relative_goal_topic, PoseStamped, self._goal_cb, queue_size=20
        )
        self.goal_valid_sub = None
        if not self.ignore_goal_valid and args.goal_valid_topic:
            self.goal_valid_sub = rospy.Subscriber(
                args.goal_valid_topic, Bool, self._goal_valid_cb, queue_size=10
            )
        self.depth_sub = rospy.Subscriber(
            args.depth_topic, Image, self._cb_depth, queue_size=10
        )

        # Used only for debug text parity with the older Twist bridge.
        self.action_to_cmd = {
            STOP: (0.0, 0.0),
            FORWARD: (args.forward_speed_debug, 0.0),
            LEFT: (0.0, args.turn_speed_debug),
            RIGHT: (0.0, -args.turn_speed_debug),
        }

        if self.debug_depth:
            os.makedirs(self.debug_depth_dump_dir, exist_ok=True)
            rospy.loginfo("Depth debug dump dir: %s", self.debug_depth_dump_dir)
        if self.replay_dump_enabled:
            os.makedirs(self.replay_dump_dir, exist_ok=True)
            rospy.loginfo("Policy replay dump dir: %s", self.replay_dump_dir)

        rospy.loginfo("Falcon nav_bridge bridge started.")
        rospy.loginfo("Subscribe depth: %s", args.depth_topic)
        rospy.loginfo("Subscribe goal:  %s", args.relative_goal_topic)
        if self.ignore_goal_valid:
            rospy.logwarn("goal_valid checking disabled by --ignore_goal_valid.")
        else:
            rospy.loginfo("Subscribe valid: %s", args.goal_valid_topic)
        rospy.loginfo("Publish command: %s", args.command_topic)

        self.watchdog = rospy.Timer(rospy.Duration(0.05), self._watchdog_cb)

    def _goal_valid_cb(self, msg: Bool):
        self.goal_valid = bool(msg.data)
        if not self.goal_valid:
            self._publish_stop()

    def _goal_cb(self, goal_msg: PoseStamped):
        self.latest_goal_msg = goal_msg
        self.goal_buffer.append(goal_msg)

    def _pick_goal_for_stamp(self, target_stamp: rospy.Time) -> Optional[PoseStamped]:
        if len(self.goal_buffer) == 0:
            return None

        if target_stamp == rospy.Time():
            return self.goal_buffer[-1]

        best = None
        best_dt = None
        for msg in self.goal_buffer:
            if msg.header.stamp == rospy.Time():
                continue
            dt = abs((target_stamp - msg.header.stamp).to_sec())
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best = msg

        if best is None:
            return self.goal_buffer[-1]
        if best_dt is not None and best_dt > self.max_goal_age_sec:
            return None
        return best

    def _relative_goal_to_polar(self, goal_msg: PoseStamped) -> Tuple[np.float32, np.float32]:
        x = float(goal_msg.pose.position.x)
        y = float(goal_msg.pose.position.y)
        z = float(goal_msg.pose.position.z)

        if self.goal_distance_mode == "3d":
            r = math.sqrt(x * x + y * y + z * z)
        else:
            r = math.sqrt(x * x + y * y)
        theta = math.atan2(y, x)
        return np.float32(r), np.float32(theta)

    def _depth_msg_to_norm_depth(
        self, depth_msg: Image
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        debug = {
            "encoding": depth_msg.encoding,
            "raw_shape": None,
            "raw_dtype": None,
            "raw_unit": "m",
            "raw_stats": None,
            "depth_m_stats": None,
            "crop_shape": None,
            "crop_stats": None,
            "norm_shape": None,
            "norm_dtype": None,
            "norm_stats": None,
        }

        if depth_msg.encoding == "16UC1":
            raw_depth = self._ros_image_to_numpy(depth_msg)
            debug["raw_shape"] = tuple(raw_depth.shape)
            debug["raw_dtype"] = str(raw_depth.dtype)
            debug["raw_unit"] = "mm"
            debug["raw_stats"] = self._depth_stats(raw_depth)
            depth_m = raw_depth.astype(np.float32) * 0.001
        elif depth_msg.encoding == "32FC1":
            raw_depth = self._ros_image_to_numpy(depth_msg)
            debug["raw_shape"] = tuple(raw_depth.shape)
            debug["raw_dtype"] = str(raw_depth.dtype)
            debug["raw_unit"] = self.depth_32fc1_unit
            debug["raw_stats"] = self._depth_stats(raw_depth)
            depth_m = raw_depth.astype(np.float32)
            if self.depth_32fc1_unit == "mm":
                depth_m = depth_m * 0.001
        else:
            raise ValueError("Unsupported image encoding: {}".format(depth_msg.encoding))

        depth_m = np.nan_to_num(
            depth_m,
            nan=self.max_depth_m,
            posinf=self.max_depth_m,
            neginf=0.0,
        )
        depth_m = np.clip(depth_m, 0.0, self.max_depth_m)
        depth_m[depth_m == 0.0] = self.max_depth_m
        debug["depth_m_stats"] = self._depth_stats(depth_m)

        depth_m = self._center_crop_to_square(depth_m)
        debug["crop_shape"] = tuple(depth_m.shape)
        debug["crop_stats"] = self._depth_stats(depth_m)

        depth_norm = depth_m / self.max_depth_m
        depth_norm = cv2.resize(
            depth_norm,
            (self.resolution, self.resolution),
            interpolation=cv2.INTER_NEAREST,
        )
        depth_norm = np.expand_dims(depth_norm.astype(np.float32), axis=-1)
        debug["norm_shape"] = tuple(depth_norm.shape)
        debug["norm_dtype"] = str(depth_norm.dtype)
        debug["norm_stats"] = self._depth_stats(depth_norm)

        self._maybe_dump_depth_sample_once(
            raw_depth=raw_depth,
            depth_m=depth_m,
            depth_norm=depth_norm,
            depth_debug=debug,
        )
        return depth_norm, debug

    def _build_obs(
        self, depth_msg: Image, goal_msg: PoseStamped
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
        r, theta = self._relative_goal_to_polar(goal_msg)
        depth_norm, depth_debug = self._depth_msg_to_norm_depth(depth_msg)
        obs = {
            self.depth_key: depth_norm,
            self.goal_key: np.array([r, theta], dtype=np.float32),
        }
        return obs, depth_debug

    def _debug_print_once(
        self,
        obs: Dict[str, np.ndarray],
        act_id: int,
        theta_in: float,
        probs: Optional[np.ndarray],
        depth_debug: Dict[str, object],
    ):
        g = obs[self.goal_key]
        d = obs[self.depth_key]
        name = self._action_name(act_id)

        if self.debug_depth:
            rospy.loginfo(
                "[DBG_DEPTH] ros(enc=%s, raw=%s %s, unit=%s) raw_stats=[%s] "
                "depth_m_stats=[%s] crop(shape=%s, stats=[%s]) "
                "falcon_input(actual=%s %s) norm_stats=[%s]",
                depth_debug["encoding"],
                depth_debug["raw_shape"],
                depth_debug["raw_dtype"],
                depth_debug["raw_unit"],
                self._fmt_stats(depth_debug["raw_stats"]),
                self._fmt_stats(depth_debug["depth_m_stats"]),
                depth_debug["crop_shape"],
                self._fmt_stats(depth_debug["crop_stats"]),
                depth_debug["norm_shape"],
                depth_debug["norm_dtype"],
                self._fmt_stats(depth_debug["norm_stats"]),
            )

        rospy.loginfo(
            "[DBG_ACT] goal[r,theta]=[%.3f, %.3f] input_theta=%.3f "
            "depth_shape=%s depth[min,max]=[%.3f,%.3f] "
            "act_id=%d(%s) publish=UInt8(%d) probs=[%s]",
            float(g[0]),
            float(g[1]),
            float(theta_in),
            tuple(d.shape),
            float(d.min()),
            float(d.max()),
            int(act_id),
            name,
            int(act_id),
            self._fmt_action_probs(probs),
        )

    @staticmethod
    def _action_name(action_id: int) -> str:
        return {
            STOP: "STOP",
            FORWARD: "FORWARD",
            LEFT: "LEFT",
            RIGHT: "RIGHT",
        }.get(int(action_id), "UNKNOWN")

    def _maybe_dump_replay_sample(
        self,
        obs: Dict[str, np.ndarray],
        act_id: int,
        probs: Optional[np.ndarray],
        recurrent_input: Dict[str, np.ndarray],
        depth_debug: Dict[str, object],
        depth_msg: Image,
        goal_msg: PoseStamped,
    ) -> bool:
        if not self.replay_dump_enabled:
            return False
        if self.replay_dump_limit >= 0 and self._replay_dump_count >= self.replay_dump_limit:
            self._replay_dump_limit_reached = True
            return True

        stamp_ns = int(rospy.Time.now().to_nsec())
        prefix = "falcon_nav_replay_{}".format(stamp_ns)
        npz_path = os.path.abspath(os.path.join(self.replay_dump_dir, prefix + ".npz"))
        meta_path = os.path.abspath(os.path.join(self.replay_dump_dir, prefix + ".json"))
        r, theta = self._relative_goal_to_polar(goal_msg)

        try:
            np.savez_compressed(
                npz_path,
                depth=obs[self.depth_key],
                depth_meter=obs[self.depth_key] * np.float32(self.max_depth_m),
                goal=obs[self.goal_key],
                hidden_in=recurrent_input["hidden_in"],
                prev_action_in=recurrent_input["prev_action_in"],
                not_done_mask_in=recurrent_input["not_done_mask_in"],
                action=np.array([act_id], dtype=np.int64),
                probs=np.array([] if probs is None else probs, dtype=np.float32),
            )
            meta = {
                "prefix": prefix,
                "obs_npz": npz_path,
                "depth_key": self.depth_key,
                "goal_key": self.goal_key,
                "resolution": int(self.resolution),
                "max_depth_m": float(self.max_depth_m),
                "deterministic": bool(self.deterministic),
                "action_id": int(act_id),
                "action_name": self._action_name(act_id),
                "depth_msg_stamp": depth_msg.header.stamp.to_sec(),
                "relative_goal_stamp": goal_msg.header.stamp.to_sec(),
                "relative_goal_frame": goal_msg.header.frame_id,
                "relative_goal_x": float(goal_msg.pose.position.x),
                "relative_goal_y": float(goal_msg.pose.position.y),
                "relative_goal_z": float(goal_msg.pose.position.z),
                "polar_r": float(r),
                "polar_theta": float(theta),
                "action_probs": None if probs is None else probs.tolist(),
                "depth_debug": depth_debug,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            self._replay_dump_count += 1
            rospy.loginfo("[REPLAY_DUMP] saved %s", meta_path)
            if self.replay_dump_limit >= 0 and self._replay_dump_count >= self.replay_dump_limit:
                self._replay_dump_limit_reached = True
                return True
        except Exception as e:
            rospy.logerr_throttle(1.0, "Policy replay dump failed: %s", str(e))
        return False

    def _publish_command(self, action_id: int):
        msg = UInt8()
        msg.data = int(action_id)
        self.command_pub.publish(msg)

    def _publish_stop(self):
        self._publish_command(STOP)

    def _process_one(self, depth_msg: Image):
        if not self.ignore_goal_valid and not self.goal_valid:
            rospy.logwarn_throttle(2.0, "goal_valid is false; publish STOP.")
            self._publish_stop()
            return

        goal_msg = self._pick_goal_for_stamp(depth_msg.header.stamp)
        if goal_msg is None:
            rospy.logwarn_throttle(2.0, "No temporally aligned relative goal received yet.")
            self._publish_stop()
            return

        try:
            _, theta_in = self._relative_goal_to_polar(goal_msg)
            obs, depth_debug = self._build_obs(depth_msg=depth_msg, goal_msg=goal_msg)
            act_id, probs, recurrent_input = self._infer_action(obs)
            if self.debug_mapping or self.debug_depth:
                self._debug_print_once(obs, act_id, float(theta_in), probs, depth_debug)
            replay_limit_reached = self._maybe_dump_replay_sample(
                obs=obs,
                act_id=act_id,
                probs=probs,
                recurrent_input=recurrent_input,
                depth_debug=depth_debug,
                depth_msg=depth_msg,
                goal_msg=goal_msg,
            )
            if replay_limit_reached:
                self._publish_stop()
                self.last_obs_time = rospy.Time.now()
                self._emit_heartbeat()
                rospy.loginfo(
                    "[REPLAY_DUMP] reached limit %d, stopping bridge.",
                    self.replay_dump_limit,
                )
                rospy.signal_shutdown("Replay dump limit reached.")
                return
            self._publish_command(act_id)
            self.last_obs_time = rospy.Time.now()
            self._emit_heartbeat()
        except Exception as e:
            self._publish_stop()
            rospy.logerr_throttle(1.0, "Falcon nav_bridge callback failed: %s", str(e))

    def _cb_depth(self, depth_msg: Image):
        self._process_one(depth_msg=depth_msg)

    def _watchdog_cb(self, _event):
        if self.last_obs_time == rospy.Time(0):
            return
        dt = (rospy.Time.now() - self.last_obs_time).to_sec()
        if dt > self.data_timeout_sec:
            self._publish_stop()
            rospy.logwarn_throttle(
                1.0,
                "Input timeout %.3fs > %.3fs, publish STOP.",
                dt,
                self.data_timeout_sec,
            )


def parse_args():
    p = argparse.ArgumentParser(
        description="RealSense depth + nav_bridge relative goal -> Falcon -> UInt8 command"
    )
    p.add_argument("--checkpoint", type=str, required=True)

    p.add_argument("--depth_topic", type=str, default="/camera/aligned_depth_to_color/image_raw")
    p.add_argument("--relative_goal_topic", type=str, default="/nav_bridge/relative_goal")
    p.add_argument("--goal_valid_topic", type=str, default="/nav_bridge/goal_valid")
    p.add_argument("--command_topic", type=str, default="/nav_bridge/discrete_cmd")
    p.add_argument("--debug_obs_topic", type=str, default="/falcon_nav_bridge/obs_heartbeat")
    p.add_argument(
        "--ignore_goal_valid",
        action="store_true",
        help="Run without subscribing/enforcing /nav_bridge/goal_valid.",
    )

    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max_depth_m", type=float, default=10.0)
    p.add_argument(
        "--depth_32fc1_unit",
        type=str,
        default="m",
        choices=["m", "mm"],
        help="Unit for 32FC1 depth images. RealSense 16UC1 is always treated as mm.",
    )
    p.add_argument(
        "--goal_distance_mode",
        type=str,
        default="planar",
        choices=["planar", "3d"],
        help="Use sqrt(x^2+y^2) or sqrt(x^2+y^2+z^2) for pointgoal distance.",
    )

    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_recurrent_layers", type=int, default=2)
    p.add_argument("--backbone", type=str, default="resnet50")
    p.add_argument("--rnn_type", type=str, default="LSTM")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--strict_checkpoint", action="store_true")

    p.add_argument("--debug_mapping", action="store_true")
    p.add_argument("--debug_depth", action="store_true")
    p.add_argument(
        "--debug_depth_dump_dir",
        type=str,
        default="./test_modules/test_results/nav_bridge_depth_samples",
    )
    p.add_argument("--replay_dump_enabled", action="store_true")
    p.add_argument(
        "--replay_dump_dir",
        type=str,
        default="./test_modules/test_results/nav_bridge_policy_replay",
    )
    p.add_argument("--replay_dump_limit", type=int, default=20)

    p.add_argument("--data_timeout_sec", type=float, default=0.3)
    p.add_argument("--max_goal_age_sec", type=float, default=0.12)
    p.add_argument("--goal_buffer_size", type=int, default=100)

    p.add_argument("--depth_obs_key", type=str, default="articulated_agent_jaw_depth")
    p.add_argument("--goal_obs_key", type=str, default="pointgoal_with_gps_compass")

    p.add_argument("--forward_speed_debug", type=float, default=0.3)
    p.add_argument("--turn_speed_debug", type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    node = FalconNavBridge(parse_args())
    node.spin()
