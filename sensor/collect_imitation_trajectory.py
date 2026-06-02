#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect real-robot trajectories for imitation learning.

Inputs:
  - Depth image from sensor/realsense_stream.py
  - Polar goal from sensor/polar_goal_tracker.py

Outputs:
  - One compressed npz per recorded step
  - manifest.jsonl with metadata for each step
  - run_meta.json with the action map and collection settings

Action ids match FalconRosBridge:
  0 stop
  1 forward
  2 turn left
  3 turn right
"""

import argparse
from collections import deque
import json
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import PointStamped, Twist
from sensor_msgs.msg import Image


ACTION_NAMES = {
    0: "stop",
    1: "forward",
    2: "left",
    3: "right",
}


def _stamp_to_sec(stamp: rospy.Time) -> float:
    if stamp == rospy.Time():
        return 0.0
    return float(stamp.to_sec())


class ImitationTrajectoryCollector(object):
    def __init__(self, args):
        rospy.init_node("imitation_trajectory_collector", anonymous=False)

        self.args = args
        self.resolution = int(args.resolution)
        self.max_depth_m = float(args.max_depth_m)
        self.control_mode = args.control_mode
        self.sample_limit = int(args.sample_limit)
        self.min_sample_interval = 0.0
        if args.sample_rate_hz > 0.0:
            self.min_sample_interval = 1.0 / float(args.sample_rate_hz)

        self.latest_polar_msg = None
        self.polar_buffer = deque(maxlen=max(10, int(args.polar_buffer_size)))
        self.latest_cmd_msg = None
        self.latest_cmd_time = rospy.Time(0)
        self.last_sample_time = rospy.Time(0)
        self.last_obs_time = rospy.Time(0)
        self.sample_count = 0
        self.goal_reached_sample_count = 0
        self.goal_reached_distance = args.goal_reached_distance
        if self.goal_reached_distance is None:
            self.goal_reached_distance = float(args.target_distance)

        self.forward_speed = float(args.forward_speed)
        self.turn_speed = float(args.turn_speed)
        self.action_to_cmd = {
            0: (0.0, 0.0),
            1: (self.forward_speed, 0.0),
            2: (0.0, self.turn_speed),
            3: (0.0, -self.turn_speed),
        }

        self.run_dir = self._prepare_output_dir(args.output_dir, args.run_id)
        self.samples_dir = os.path.join(self.run_dir, "samples")
        os.makedirs(self.samples_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.run_dir, "manifest.jsonl")
        self.manifest_f = open(self.manifest_path, "a", encoding="utf-8")
        rospy.on_shutdown(self._on_shutdown)
        self._write_run_meta()

        self.cmd_pub = None
        if self.control_mode == "heuristic":
            self.cmd_pub = rospy.Publisher(args.cmd_vel_topic, Twist, queue_size=10)
        else:
            self.cmd_sub = rospy.Subscriber(
                args.action_source_topic, Twist, self._cmd_cb, queue_size=20
            )

        self.polar_sub = rospy.Subscriber(
            args.polar_topic, PointStamped, self._polar_cb, queue_size=20
        )
        self.depth_sub = rospy.Subscriber(
            args.depth_topic, Image, self._depth_cb, queue_size=10
        )
        self.watchdog = rospy.Timer(rospy.Duration(0.05), self._watchdog_cb)

        rospy.loginfo("imitation_trajectory_collector started.")
        rospy.loginfo("control_mode: %s", self.control_mode)
        rospy.loginfo("Subscribe depth: %s", args.depth_topic)
        rospy.loginfo("Subscribe polar: %s", args.polar_topic)
        if self.control_mode == "heuristic":
            rospy.loginfo("Publish cmd_vel: %s", args.cmd_vel_topic)
        else:
            rospy.loginfo("Subscribe action source: %s", args.action_source_topic)
        rospy.loginfo("Output dir: %s", self.run_dir)

    @staticmethod
    def _prepare_output_dir(output_dir: str, run_id: Optional[str]) -> str:
        root = os.path.abspath(output_dir)
        if not run_id:
            run_id = rospy.Time.now().to_sec()
            run_id = "{:.6f}".format(run_id).replace(".", "_")
        run_dir = os.path.join(root, str(run_id))
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _write_run_meta(self):
        meta = {
            "depth_topic": self.args.depth_topic,
            "polar_topic": self.args.polar_topic,
            "cmd_vel_topic": self.args.cmd_vel_topic,
            "action_source_topic": self.args.action_source_topic,
            "control_mode": self.control_mode,
            "resolution": self.resolution,
            "max_depth_m": self.max_depth_m,
            "depth_shape": [self.resolution, self.resolution],
            "depth_unit": "m",
            "depth_range_m": [0.0, self.max_depth_m],
            "goal_shape": [2],
            "goal_format": ["r_m", "theta_rad"],
            "action_names": ACTION_NAMES,
            "action_to_cmd": {
                str(k): {"linear_x": v[0], "angular_z": v[1]}
                for k, v in self.action_to_cmd.items()
            },
            "heuristic": {
                "target_distance": self.args.target_distance,
                "turn_angle_thresh": self.args.turn_angle_thresh,
                "dist_deadband": self.args.dist_deadband,
            },
            "sample_rate_hz": self.args.sample_rate_hz,
            "sample_limit": self.sample_limit,
            "goal_reached_distance": self.goal_reached_distance,
            "stop_after_goal_steps": self.args.stop_after_goal_steps,
            "max_polar_age_sec": self.args.max_polar_age_sec,
        }
        path = os.path.join(self.run_dir, "run_meta.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _ros_image_to_numpy(msg: Image) -> np.ndarray:
        if msg.encoding == "16UC1":
            row_bytes = msg.width * 2
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
            arr = raw[:, :row_bytes].copy().view(np.uint16).reshape(msg.height, msg.width)
        elif msg.encoding == "32FC1":
            row_bytes = msg.width * 4
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
            arr = raw[:, :row_bytes].copy().view(np.float32).reshape(msg.height, msg.width)
        else:
            raise ValueError("Unsupported image encoding: {}".format(msg.encoding))

        if msg.is_bigendian:
            arr = arr.byteswap().newbyteorder()
        return arr

    @staticmethod
    def _center_crop_to_square(arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        return arr[y0 : y0 + side, x0 : x0 + side]

    @staticmethod
    def _depth_stats(arr: np.ndarray) -> Dict[str, float]:
        arr_f = arr.astype(np.float32, copy=False)
        finite = np.isfinite(arr_f)
        valid = arr_f[finite]
        if valid.size == 0:
            return {"valid_ratio": 0.0, "min": None, "max": None, "mean": None}
        return {
            "valid_ratio": float(valid.size) / float(arr_f.size),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "mean": float(np.mean(valid)),
        }

    def _depth_msg_to_arrays(self, msg: Image) -> Tuple[np.ndarray, Dict[str, object]]:
        raw = self._ros_image_to_numpy(msg)
        if msg.encoding == "16UC1":
            depth_m = raw.astype(np.float32) * 0.001
            raw_unit = "mm"
        else:
            depth_m = raw.astype(np.float32)
            raw_unit = "m"

        depth_m = np.nan_to_num(
            depth_m,
            nan=self.max_depth_m,
            posinf=self.max_depth_m,
            neginf=0.0,
        )
        depth_m = np.clip(depth_m, 0.0, self.max_depth_m)
        depth_m[depth_m == 0.0] = self.max_depth_m

        cropped = self._center_crop_to_square(depth_m)
        depth_meter = cv2.resize(
            cropped,
            (self.resolution, self.resolution),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)

        debug = {
            "encoding": msg.encoding,
            "raw_unit": raw_unit,
            "raw_shape": list(raw.shape),
            "raw_dtype": str(raw.dtype),
            "raw_stats": self._depth_stats(raw),
            "depth_shape": list(depth_meter.shape),
            "depth_unit": "m",
            "depth_stats": self._depth_stats(depth_meter),
        }
        return depth_meter, debug

    def _polar_cb(self, msg: PointStamped):
        self.latest_polar_msg = msg
        self.polar_buffer.append(msg)

    def _cmd_cb(self, msg: Twist):
        self.latest_cmd_msg = msg
        self.latest_cmd_time = rospy.Time.now()

    def _pick_polar_for_stamp(self, target_stamp: rospy.Time) -> Optional[PointStamped]:
        if len(self.polar_buffer) == 0:
            return None
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
        if best_dt is not None and best_dt > self.args.max_polar_age_sec:
            return None
        return best

    def _heuristic_action(self, r: float, theta: float) -> int:
        if theta > self.args.turn_angle_thresh:
            return 2
        if theta < -self.args.turn_angle_thresh:
            return 3
        if (r - self.args.target_distance) > self.args.dist_deadband:
            return 1
        return 0

    def _cmd_to_action(self, cmd: Twist) -> int:
        lin = float(cmd.linear.x)
        ang = float(cmd.angular.z)

        lin_score = lin / max(abs(self.forward_speed), 1e-6)
        ang_score = ang / max(abs(self.turn_speed), 1e-6)
        if abs(lin_score) < self.args.cmd_deadband and abs(ang_score) < self.args.cmd_deadband:
            return 0
        if abs(ang_score) > abs(lin_score):
            return 2 if ang_score > 0.0 else 3
        return 1 if lin_score > 0.0 else 0

    def _select_action(self, polar_msg: PointStamped) -> Optional[int]:
        if self.control_mode == "heuristic":
            return self._heuristic_action(
                float(polar_msg.point.x),
                float(polar_msg.point.y),
            )

        if self.latest_cmd_msg is None:
            rospy.logwarn_throttle(2.0, "No cmd_vel action source received yet; skip sample.")
            return None
        age = (rospy.Time.now() - self.latest_cmd_time).to_sec()
        if age > self.args.max_cmd_age_sec:
            rospy.logwarn_throttle(
                2.0,
                "cmd_vel action source is stale %.3fs > %.3fs; skip sample.",
                age,
                self.args.max_cmd_age_sec,
            )
            return None
        return self._cmd_to_action(self.latest_cmd_msg)

    def _publish_cmd(self, action_id: int):
        if self.cmd_pub is None:
            return
        lin, ang = self.action_to_cmd[action_id]
        msg = Twist()
        msg.linear.x = lin
        msg.angular.z = ang
        self.cmd_pub.publish(msg)

    def _publish_stop(self):
        if self.cmd_pub is not None:
            self.cmd_pub.publish(Twist())

    def _should_sample_now(self) -> bool:
        if self.sample_limit >= 0 and self.sample_count >= self.sample_limit:
            self._shutdown_after_stop("Sample limit reached.")
            return False
        if self.last_sample_time == rospy.Time(0):
            return True
        dt = (rospy.Time.now() - self.last_sample_time).to_sec()
        return dt >= self.min_sample_interval

    def _shutdown_after_stop(self, reason: str):
        self._publish_stop()
        rospy.loginfo("%s Saved samples: %d", reason, self.sample_count)
        rospy.signal_shutdown(reason)

    def _update_goal_reached_stop_condition(self, polar_msg: PointStamped):
        if self.args.stop_after_goal_steps < 0:
            return

        r = float(polar_msg.point.x)
        reached = r <= float(self.goal_reached_distance)
        if reached:
            self.goal_reached_sample_count += 1
        else:
            self.goal_reached_sample_count = 0

        if reached and self.goal_reached_sample_count >= self.args.stop_after_goal_steps:
            self._shutdown_after_stop(
                "Goal distance reached for {} samples.".format(
                    self.goal_reached_sample_count
                )
            )

    def _save_sample(
        self,
        depth_meter: np.ndarray,
        depth_debug: Dict[str, object],
        depth_msg: Image,
        polar_msg: PointStamped,
        action_id: int,
    ):
        stamp_ns = int(rospy.Time.now().to_nsec())
        prefix = "sample_{:06d}_{}".format(self.sample_count, stamp_ns)
        npz_path = os.path.abspath(os.path.join(self.samples_dir, prefix + ".npz"))
        r = float(polar_msg.point.x)
        theta = float(polar_msg.point.y)
        tag_id = float(polar_msg.point.z)
        lin, ang = self.action_to_cmd[action_id]

        np.savez_compressed(
            npz_path,
            depth=depth_meter,
            goal=np.array([r, theta], dtype=np.float32),
            action=np.array([action_id], dtype=np.int64),
            cmd=np.array([lin, ang], dtype=np.float32),
            tag_id=np.array([tag_id], dtype=np.float32),
            depth_stamp=np.array([_stamp_to_sec(depth_msg.header.stamp)], dtype=np.float64),
            polar_stamp=np.array([_stamp_to_sec(polar_msg.header.stamp)], dtype=np.float64),
        )

        record = {
            "index": self.sample_count,
            "sample_npz": npz_path,
            "action_id": int(action_id),
            "action_name": ACTION_NAMES[action_id],
            "cmd_linear_x": float(lin),
            "cmd_angular_z": float(ang),
            "polar_r": r,
            "polar_theta": theta,
            "tag_id": tag_id,
            "depth_msg_stamp": _stamp_to_sec(depth_msg.header.stamp),
            "polar_msg_stamp": _stamp_to_sec(polar_msg.header.stamp),
            "depth_debug": depth_debug,
        }
        self.manifest_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.sample_count % max(1, self.args.flush_every) == 0:
            self.manifest_f.flush()

        self.sample_count += 1
        self.last_sample_time = rospy.Time.now()
        rospy.loginfo_throttle(
            0.5,
            "recorded %d samples | r=%.3f theta=%.3f action=%d(%s) cmd=(%.2f, %.2f)",
            self.sample_count,
            r,
            theta,
            action_id,
            ACTION_NAMES[action_id],
            lin,
            ang,
        )

    def _depth_cb(self, depth_msg: Image):
        if not self._should_sample_now():
            return

        polar_msg = self._pick_polar_for_stamp(depth_msg.header.stamp)
        if polar_msg is None:
            rospy.logwarn_throttle(2.0, "No fresh polar goal received yet; publish stop.")
            self._publish_stop()
            return

        try:
            action_id = self._select_action(polar_msg)
            if action_id is None:
                return

            self._publish_cmd(action_id)
            depth_meter, depth_debug = self._depth_msg_to_arrays(depth_msg)
            self._save_sample(
                depth_meter=depth_meter,
                depth_debug=depth_debug,
                depth_msg=depth_msg,
                polar_msg=polar_msg,
                action_id=action_id,
            )
            self.last_obs_time = rospy.Time.now()
            self._update_goal_reached_stop_condition(polar_msg)
            if self.sample_limit >= 0 and self.sample_count >= self.sample_limit:
                self._shutdown_after_stop("Sample limit reached.")
        except Exception as e:
            self._publish_stop()
            rospy.logerr_throttle(1.0, "trajectory collection callback failed: %s", str(e))

    def _watchdog_cb(self, _event):
        if self.control_mode != "heuristic":
            return
        if self.last_obs_time == rospy.Time(0):
            return
        dt = (rospy.Time.now() - self.last_obs_time).to_sec()
        if dt > self.args.data_timeout_sec:
            self._publish_stop()
            rospy.logwarn_throttle(
                1.0,
                "Input timeout %.3fs > %.3fs, publish stop.",
                dt,
                self.args.data_timeout_sec,
            )

    def _on_shutdown(self):
        self._publish_stop()
        try:
            self.manifest_f.flush()
            self.manifest_f.close()
        except Exception:
            pass

    def spin(self):
        rospy.spin()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect depth+polar+discrete-action trajectories for imitation learning."
    )
    parser.add_argument("--depth_topic", type=str, default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--polar_topic", type=str, default="/tag_polar")
    parser.add_argument("--cmd_vel_topic", type=str, default="/cmd_vel")
    parser.add_argument(
        "--action_source_topic",
        type=str,
        default="/cmd_vel",
        help="Twist topic used as labels in passive mode.",
    )
    parser.add_argument(
        "--control_mode",
        type=str,
        default="heuristic",
        choices=["heuristic", "passive"],
        help="heuristic publishes discrete actions; passive records labels from action_source_topic.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_modules/test_results/il_trajectories",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument(
        "--sample_limit",
        "--max_steps",
        dest="sample_limit",
        type=int,
        default=200,
        help="Maximum number of recorded samples. -1 means no hard limit.",
    )
    parser.add_argument("--sample_rate_hz", type=float, default=10.0)
    parser.add_argument("--flush_every", type=int, default=10)

    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max_depth_m", type=float, default=10.0)
    parser.add_argument("--max_polar_age_sec", type=float, default=0.12)
    parser.add_argument("--polar_buffer_size", type=int, default=100)
    parser.add_argument("--data_timeout_sec", type=float, default=0.3)

    parser.add_argument("--forward_speed", type=float, default=0.3)
    parser.add_argument("--turn_speed", type=float, default=0.3)
    parser.add_argument("--target_distance", type=float, default=0.2)
    parser.add_argument("--turn_angle_thresh", type=float, default=0.3)
    parser.add_argument("--dist_deadband", type=float, default=0.03)
    parser.add_argument(
        "--goal_reached_distance",
        type=float,
        default=None,
        help="Distance threshold for auto-stop. Defaults to target_distance.",
    )
    parser.add_argument(
        "--stop_after_goal_steps",
        type=int,
        default=3,
        help="Stop after this many consecutive recorded samples at goal distance. -1 disables.",
    )

    parser.add_argument("--max_cmd_age_sec", type=float, default=0.5)
    parser.add_argument("--cmd_deadband", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        node = ImitationTrajectoryCollector(parse_args())
        node.spin()
    except rospy.ROSInterruptException:
        pass
